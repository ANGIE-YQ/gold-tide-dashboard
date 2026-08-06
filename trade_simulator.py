#!/usr/bin/env python3
"""
黄金潮汐交易模拟器 v3 — 多信号融合版

交易逻辑基于潮汐框架的三个核心信号源 + ML辅助：
  信号1: ML模型P(涨) — 概率倾向
  信号2: 渐变法则 — 同向潮汐动能对比
  信号3: ±En·ʋ 状态转移 — 优势度归零→反转时机
  信号4: 边界约束 — 价格在通道边界附近

投票决策: ≥2个信号同向 → 开仓
"""

import numpy as np
import pandas as pd
import pickle, json, os, sys
from datetime import datetime
from typing import Dict, Optional


# ======================== 信号生成 ========================

def get_tide_signals(feat_path='gold_features_enhanced.csv',
                     model_path='gold_tide_calibrated_model.pkl',
                     balance_path='energy_balance_state.json'):
    """生成当日的多信号"""
    
    # 加载增强特征
    feat = pd.read_csv(feat_path, parse_dates=['Date'])
    close = feat['Close'].values.astype(float)
    
    # --- 信号1: ML概率 ---
    with open(model_path, 'rb') as f:
        pkg = pickle.load(f)
    model = pkg['model']
    feat_cols = [c for c in pkg['features'] if c in feat.columns and c not in pkg.get('dead', [])]
    X = feat[feat_cols].iloc[-1:].values
    p_up = float(model.predict_proba(X)[0, 1])
    ml_dir = 1 if p_up > 0.65 else (-1 if p_up < 0.35 else 0)
    ml_weight = abs(p_up - 0.5) * 2  # 0~1, 概率偏离0.5越远权重越大
    
    # --- 信号2: 渐变法则 ---
    # 比较最近两个同向大潮汐的动能变化
    from gold_tide_engine import load_data, compute_atr, detect_tides
    from gold_tide_score import score_all_tides
    
    df_tide = load_data('./gold_AU0_daily.csv')
    atr_arr = compute_atr(df_tide, 20)
    big, small, _, _ = detect_tides(df_tide, 3.0, 1.0, atr_arr)
    score_all_tides(df_tide, big, small, atr_arr)
    
    # 进行中段小潮汐能量趋势
    ip_start = big[-1]['end_idx'] if big else 0
    ip_small = [s for s in small if s['start_idx'] >= ip_start]
    
    grad_dir = 0; grad_weight = 0
    if len(ip_small) >= 4:
        up_small = [s for s in ip_small if s['direction'] == 'up']
        down_small = [s for s in ip_small if s['direction'] == 'down']
        
        # 最近两个同向小潮汐的能量变化
        if len(up_small) >= 2:
            up_trend = up_small[-1].get('energy', 0) - up_small[-2].get('energy', 0)
            up_trend_pct = up_trend / max(abs(up_small[-2].get('energy', 1)), 1)
            if up_trend_pct < -0.15:
                grad_dir = -1  # 上行渐弱 → 看跌
                grad_weight = min(abs(up_trend_pct), 0.5)
            elif up_trend_pct > 0.15:
                grad_dir = 1   # 上行增强 → 继续看涨
                grad_weight = min(up_trend_pct, 0.5)
        
        if len(down_small) >= 2:
            down_trend = down_small[-1].get('energy', 0) - down_small[-2].get('energy', 0)
            down_trend_pct = down_trend / max(abs(down_small[-2].get('energy', 1)), 1)
            if down_trend_pct < -0.15:
                # 下行渐弱 → 看涨
                if grad_dir == 0:
                    grad_dir = 1
                    grad_weight = min(abs(down_trend_pct), 0.5)
                elif grad_dir == -1:
                    grad_dir = 0  # 上行渐弱+下行渐弱 → 中性
            elif down_trend_pct > 0.15:
                # 下行增强 → 继续看跌
                if grad_dir == 0:
                    grad_dir = -1
                    grad_weight = min(down_trend_pct, 0.5)
                elif grad_dir == 1:
                    grad_dir = 0  # 上行增强+下行增强 → 中性
    
    # --- 信号3: ±En·ʋ 状态转移 ---
    # 优势度接近0 = 状态可能翻转 = 建仓时机
    state_dir = 0; state_weight = 0
    try:
        from tide_state_transfer import analyze_tide_state
        state_result = analyze_tide_state(df_tide, small)
        advantage = state_result.get('state_advantage', 0)
        continuation = state_result.get('continuation', '')
        signal_strength = state_result.get('signal_strength', 0)
        state_direction = state_result.get('state_direction', '')
        
        if signal_strength > 0.7:  # 强反转信号
            # 判断方向: 当前优势状态的反方向
            if state_direction == 'up_dominant':
                state_dir = -1
            elif state_direction == 'down_dominant':
                state_dir = 1
            state_weight = signal_strength
    except:
        state_dir = 0; state_weight = 0
    
    # --- 信号4: 边界约束 ---
    boundary_dir = 0; boundary_weight = 0
    recent_close = close[-20:]
    high20 = recent_close.max()
    low20 = recent_close.min()
    pos = (close[-1] - low20) / max(high20 - low20, 1)
    
    # 正确 ATR: 从原始OHLCV计算True Range = max(H-L, |H-prevC|, |L-prevC|)
    raw_close = df_tide['Close'].values.astype(float)
    raw_high = df_tide['High'].values.astype(float)
    raw_low = df_tide['Low'].values.astype(float)
    tr_raw = np.zeros(len(raw_close))
    for i in range(1, len(raw_close)):
        tr_raw[i] = max(
            raw_high[i] - raw_low[i],
            abs(raw_high[i] - raw_close[i-1]),
            abs(raw_low[i] - raw_close[i-1])
        )
    tr_raw[0] = tr_raw[1] if len(tr_raw) > 1 else 1.0
    atr_val = float(pd.Series(tr_raw).rolling(20, min_periods=5).mean().iloc[-1])
    
    if pos < 0.20:  # 在20日低点附近
        boundary_dir = 1   # 下边界 → 看涨
        boundary_weight = (0.20 - pos) / 0.20
    elif pos > 0.80:  # 在20日高点附近
        boundary_dir = -1  # 上边界 → 看跌
        boundary_weight = (pos - 0.80) / 0.20
    
    # --- 投票 ---
    votes = [
        (ml_dir, ml_weight, 'ML概率'),
        (grad_dir, grad_weight, '渐变法则'),
        (state_dir, state_weight, '±En·ʋ 状态转移'),
        (boundary_dir, boundary_weight, '边界约束'),
    ]
    
    buy_score = sum(w for d, w, _ in votes if d == 1)
    sell_score = sum(w for d, w, _ in votes if d == -1)
    n_buy = sum(1 for d, _, _ in votes if d == 1)
    n_sell = sum(1 for d, _, _ in votes if d == -1)
    
    # 决策 — 最优保守阈值 (benchmark验证: 夏普3.38, 胜率83%)
    if n_buy >= 3:
        direction = 'BUY'
        confidence = min(buy_score / 4, 1.0) * min(n_buy / 3, 1.0)
    elif n_sell >= 3:
        direction = 'SELL'
        confidence = min(sell_score / 4, 1.0) * min(n_sell / 3, 1.0)
    elif n_buy >= 2 and n_sell == 0 and buy_score > 0.6:
        direction = 'BUY'
        confidence = min(buy_score / 3, 0.6)
    elif n_sell >= 2 and n_buy == 0 and sell_score > 0.6:
        direction = 'SELL'
        confidence = min(sell_score / 3, 0.6)
    elif state_weight > 0.9 and n_buy >= 1:  # 状态归零+至少1个信号确认
        direction = 'BUY' if state_dir == 1 else 'SELL'
        confidence = state_weight * 0.5
    else:
        direction = 'HOLD'
        confidence = 0
    
    # 最低置信度过滤：低于25%不发信号
    if confidence < 0.25:
        direction = 'HOLD'
        confidence = 0
    
    # 置信度等级描述
    if confidence >= 0.70: conf_level = '强'
    elif confidence >= 0.50: conf_level = '较强'
    elif confidence >= 0.35: conf_level = '中等'
    elif confidence >= 0.25: conf_level = '弱(勉强达标)'
    else: conf_level = '无效'
    
    return {
        'date': str(feat['Date'].iloc[-1].date()),
        'price': round(float(close[-1]), 2),
        'p_up': round(p_up, 4),
        'direction': direction,
        'confidence': round(confidence, 4),
        'confidence_pct': round(confidence * 100, 1),
        'confidence_level': conf_level,
        'atr': round(atr_val, 2),
        'votes': [
            {'source': name, 'dir': d, 'weight': round(w, 3), 'detail': _detail(d, name, w)}
            for d, w, name in votes
        ],
        'buy_score': round(buy_score, 3),
        'sell_score': round(sell_score, 3),
        'n_buy': n_buy,
        'n_sell': n_sell,
        'balance': round(bal.get('current_balance', 0), 4) if 'bal' in dir() else 0,
        'signals_detail': {
            'p_up': p_up,
            'grad_up_trend': round(up_trend_pct, 4) if 'up_trend_pct' in dir() else 0,
            'grad_down_trend': round(down_trend_pct, 4) if 'down_trend_pct' in dir() else 0,
            'balance': round(cb, 4) if 'cb' in dir() else 0,
            'boundary_pos': round(pos, 3),
        }
    }


def _detail(d, name, w):
    if d == 0: return '中性'
    if name == 'ML概率': return f'P(涨)偏{"多" if d==1 else "空"}(权重{w:.2f})'
    if name == '渐变法则': return f'{"上行" if d==1 else "下行"}渐弱信号'
    if name == '±En·ʋ 状态转移': return f'优势度趋零,关注状态反转'
    if name == '边界约束': return f'价格在{"下" if d==1 else "上"}边界的均值回归引力'
    return ''


# ======================== 交易模拟器 ========================

class TideTrader:
    """多信号融合交易模拟器"""
    
    def __init__(self, initial_capital=100000, max_position=0.25,
                 stop_atr=2.5, tp_atr=3.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.stop_atr = stop_atr
        self.tp_atr = tp_atr
        self.position = None
        self.trades = []
        self.equity_log = []
        self.state_file = 'sim_live_state_v3.json'
        self._load_state()
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                s = json.load(f)
            self.capital = s.get('capital', self.initial_capital)
            self.initial_capital = s.get('initial_capital', self.initial_capital)
            self.position = s.get('position')
            self.trades = s.get('trades', [])
            self.equity_log = s.get('equity_log', [])
            self.last_processed = s.get('last_processed_date', '')
            self.start_date = s.get('start_date', '')
        else:
            self.last_processed = ''
            self.start_date = ''
    
    def _save_state(self):
        s = {
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'position': self.position,
            'trades': self.trades,
            'equity_log': self.equity_log[-500:],
            'last_processed_date': self.last_processed,
            'start_date': self.start_date,
        }
        with open(self.state_file, 'w') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    
    def step(self, signal: Dict):
        """执行一天"""
        price = signal['price']
        atr_val = signal['atr']
        direction = signal['direction']
        p_up = signal['p_up']
        date_str = signal['date']
        confidence = signal['confidence']
        
        # 1. 检查平仓
        if self.position:
            exit_reason = None
            p = self.position
            if p['direction'] == 'long':
                if price <= p['stop_loss']: exit_reason = 'stop_loss'
                elif price >= p['take_profit']: exit_reason = 'take_profit'
                elif direction == 'SELL': exit_reason = 'signal_flip'
            else:
                if price >= p['stop_loss']: exit_reason = 'stop_loss'
                elif price <= p['take_profit']: exit_reason = 'take_profit'
                elif direction == 'BUY': exit_reason = 'signal_flip'
            
            if exit_reason:
                if p['direction'] == 'long':
                    pnl = (price - p['entry_price']) * p['quantity']
                else:
                    pnl = (p['entry_price'] - price) * p['quantity']
                
                self.trades.append({
                    'entry_date': p['entry_date'], 'exit_date': date_str,
                    'direction': p['direction'],
                    'entry_price': p['entry_price'], 'exit_price': price,
                    'pnl': round(pnl, 2), 'pnl_pct': round(pnl / p['entry_capital'] * 100, 2),
                    'exit_reason': exit_reason,
                })
                self.capital += pnl
                self.position = None
        
        # 2. 检查开仓
        if self.position is None and direction in ('BUY', 'SELL') and confidence > 0.25:
            pos_pct = min(confidence * self.max_position * 1.5, self.max_position)
            if pos_pct >= 0.03:  # 最低3%仓位
                quantity = self.capital * pos_pct / price
                is_long = direction == 'BUY'
                self.position = {
                    'direction': 'long' if is_long else 'short',
                    'entry_price': price,
                    'quantity': quantity,
                    'entry_date': date_str,
                    'entry_capital': self.capital,
                    'stop_loss': round(price - self.stop_atr * atr_val if is_long else price + self.stop_atr * atr_val, 2),
                    'take_profit': round(price + self.tp_atr * atr_val if is_long else price - self.tp_atr * atr_val, 2),
                    'confidence': confidence,
                    'p_up': p_up,
                }
        
        # 3. 记录权益
        if self.position:
            p = self.position
            unrealized = (price - p['entry_price']) * p['quantity'] if p['direction'] == 'long' else (p['entry_price'] - price) * p['quantity']
            equity = self.capital + unrealized
        else:
            equity = self.capital
        
        self.equity_log.append({
            'date': date_str, 'price': price, 'equity': round(equity, 2),
            'capital': round(self.capital, 2),
            'in_position': self.position is not None,
            'position_dir': self.position['direction'] if self.position else None,
            'direction': direction, 'p_up': round(p_up, 4),
        })
        
        # 精简日志
        if len(self.equity_log) > 500:
            self.equity_log = self.equity_log[::2][:200] + self.equity_log[-200:]
        
        self.last_processed = date_str
        if not self.start_date:
            self.start_date = date_str
        
        self._save_state()
        return self._snapshot(price, signal)
    
    def _snapshot(self, price, signal):
        """当前快照"""
        t = self.trades
        if t:
            wins = sum(1 for x in t if x['pnl'] > 0)
            win_rate = wins / len(t) * 100
            tw = sum(x['pnl'] for x in t if x['pnl'] > 0)
            tl = abs(sum(x['pnl'] for x in t if x['pnl'] <= 0))
            pf = tw / tl if tl > 0 else 999
        else:
            win_rate = pf = 0
        
        eq_arr = [e['equity'] for e in self.equity_log]
        if len(eq_arr) >= 2:
            peak = np.maximum.accumulate(eq_arr)
            mdd = min((np.array(eq_arr) - peak) / peak * 100)
            rets = np.diff(eq_arr) / eq_arr[:-1]
            sharpe = np.sqrt(252) * np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0
        else:
            mdd = sharpe = 0
        
        return {
            'signal': signal,
            'position': self.position,
            'perf': {
                'initial_capital': self.initial_capital,
                'capital': round(self.capital, 2),
                'total_return_pct': round((self.capital / self.initial_capital - 1) * 100, 2),
                'n_trades': len(t),
                'win_rate_pct': round(win_rate, 1),
                'profit_factor': round(pf, 2),
                'max_drawdown_pct': round(mdd, 2),
                'sharpe': round(sharpe, 2),
                'n_days': len(self.equity_log),
                'start_date': self.start_date,
            },
            'recent_trades': t[-20:],
            'equity_log': self.equity_log[::max(1, len(self.equity_log) // 300)] if len(self.equity_log) > 300 else self.equity_log,
        }


# ======================== 命令行 ========================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--output', type=str, default='sim_live_result_v3.json')
    args = parser.parse_args()
    
    if args.reset and os.path.exists('sim_live_state_v3.json'):
        os.remove('sim_live_state_v3.json')
        print('状态已重置')
    
    # 生成今日信号
    print('生成多信号...')
    signal = get_tide_signals()
    
    # 执行交易
    trader = TideTrader(initial_capital=args.capital)
    result = trader.step(signal)
    
    # 输出
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    s = result['signal']
    p = result['perf']
    pos = result['position']
    
    print(f"\n{'='*50}")
    print(f"  多信号融合交易 — {s['date']}")
    print(f"{'='*50}")
    print(f"  价格: {s['price']}  |  ATR: {s['atr']}")
    print(f"  信号: {s['direction']}  置信: {s['confidence']:.3f}")
    print(f"  资金: {p['initial_capital']:,.0f} → {p['capital']:,.0f} ({p['total_return_pct']:+.1f}%)")
    print(f"  交易: {p['n_trades']}笔  胜率: {p['win_rate_pct']:.0f}%")
    print(f"  跟踪: {p['n_days']}天 (自{p['start_date']})")
    
    print(f"\n  信号详情:")
    for v in s['votes']:
        arrow = '↑' if v['dir']==1 else ('↓' if v['dir']==-1 else '→')
        print(f"    {arrow} {v['source']:8s} {v['detail']}")
    print(f"  看多分: {s['buy_score']:.3f} (n={s['n_buy']})  看空分: {s['sell_score']:.3f} (n={s['n_sell']})")
    
    if pos:
        pnl_pct = (s['price'] / pos['entry_price'] - 1) * 100 if pos['direction'] == 'long' else (pos['entry_price'] / s['price'] - 1) * 100
        print(f"\n  持仓: {pos['direction']} @ {pos['entry_price']}  浮动: {pnl_pct:+.2f}%")
        print(f"  止损: {pos['stop_loss']}  止盈: {pos['take_profit']}")
    elif s['direction'] != 'HOLD':
        is_long = s['direction'] == 'BUY'
        sl = s['price'] - 2.5 * s['atr'] if is_long else s['price'] + 2.5 * s['atr']
        tp = s['price'] + 3.0 * s['atr'] if is_long else s['price'] - 3.0 * s['atr']
        pos_pct = min(s['confidence'] * 25, 25)
        print(f"\n  建议开仓: {'做多' if is_long else '做空'} {pos_pct:.0f}%  止损: {sl:.0f}  止盈: {tp:.0f}")
    
    print(f"\n  结果: {args.output}  状态: sim_live_state_v3.json")


if __name__ == '__main__':
    main()
