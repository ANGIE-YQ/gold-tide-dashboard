#!/usr/bin/env python3
"""
黄金潮汐模型 — 交易模拟器 v2（实盘跟踪版）
每天自动推进：读取最新模型信号 → 检查持仓 → 开平仓 → 更新绩效 → 持久化状态
"""

import numpy as np
import pandas as pd
import pickle, json, os, sys
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict


# ======================== 数据模型 ========================

@dataclass
class Position:
    direction: str   # 'long' / 'short'
    entry_price: float
    quantity: float
    entry_date: str
    entry_signal_p: float
    stop_loss: float
    take_profit: float
    entry_capital: float  # 开仓时总资金

@dataclass
class ClosedTrade:
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float      # vs 开仓时总资金
    exit_reason: str
    signal_p: float

@dataclass
class SimState:
    capital: float
    initial_capital: float
    position: Optional[dict] = None       # Position as dict for JSON
    trades: list = field(default_factory=list)    # list of ClosedTrade dicts
    equity_log: list = field(default_factory=list)  # daily snapshots
    last_processed_date: str = ''
    start_date: str = ''


# ======================== 核心 — 实盘模拟引擎 ========================

class LiveTrader:
    """
    实盘模拟器：状态持久化，每次运行自动推进到最新数据日
    
    规则：
    - 信号 P(涨) > 0.60 → 做多, P < 0.40 → 做空
    - 仓位: Kelly公式 × 置信度, 上限25%
    - 止损: 2.5×ATR, 止盈: 3.0×ATR
    - 信号反转时平仓
    """
    
    def __init__(self, initial_capital=100000, max_position=0.25,
                 stop_atr=2.5, tp_atr=3.0, buy_thr=0.60, sell_thr=0.40,
                 state_path='sim_live_state.json',
                 feat_path='gold_features_enhanced.csv',
                 model_path='gold_tide_calibrated_model.pkl'):
        
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.stop_atr = stop_atr
        self.tp_atr = tp_atr
        self.buy_thr = buy_thr
        self.sell_thr = sell_thr
        self.state_path = state_path
        self.feat_path = feat_path
        self.model_path = model_path
        
        # 加载或初始化状态
        self.state = self._load_state()
        
        # 加载模型
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
        self.calibrated = pkg['model']
        self.feature_cols = [c for c in pkg['features'] 
                            if c not in pkg.get('dead', [])]
    
    def _load_state(self) -> SimState:
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            return SimState(
                capital=d.get('capital', self.initial_capital),
                initial_capital=d.get('initial_capital', self.initial_capital),
                position=d.get('position'),
                trades=d.get('trades', []),
                equity_log=d.get('equity_log', []),
                last_processed_date=d.get('last_processed_date', ''),
                start_date=d.get('start_date', ''),
            )
        return SimState(capital=self.initial_capital, initial_capital=self.initial_capital)
    
    def _save_state(self):
        d = asdict(self.state)
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    
    def _kelly_pct(self, p_win: float) -> float:
        """半凯利仓位"""
        win_loss_ratio = 1.5
        p_lose = 1 - p_win
        kelly = max(0, (p_win * win_loss_ratio - p_lose) / win_loss_ratio)
        return min(kelly * 0.5, self.max_position)
    
    def _compute_atr(self, close, high=None, low=None, period=20):
        """计算ATR"""
        if high is None: high = close
        if low is None: low = close
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1]))
        )
        tr = np.concatenate([[tr[0]], tr])
        return pd.Series(tr).rolling(period, min_periods=5).mean().values
    
    def run_daily(self) -> Dict:
        """
        每日执行：加载最新特征 → 获取当天信号 → 检查/执行交易 → 保存状态
        返回今日快照（供看板展示）
        """
        # 加载特征数据
        feat = pd.read_csv(self.feat_path, parse_dates=['Date'])
        feat['Date'] = feat['Date'].dt.date
        
        # 过滤有效特征列
        avail_cols = [c for c in self.feature_cols if c in feat.columns]
        X_all = feat[avail_cols].values
        
        # 获取模型预测概率
        probs = self.calibrated.predict_proba(X_all)[:, 1]
        
        # 价格、日期、ATR
        close = feat['Close'].values.astype(float)
        dates = feat['Date'].values
        high = feat['High'].values.astype(float) if 'High' in feat.columns else close
        low = feat['Low'].values.astype(float) if 'Low' in feat.columns else close
        atr = self._compute_atr(close, high, low)
        
        # 确定需要处理的日期范围
        if not self.state.last_processed_date:
            # 首次运行：从最近60天开始（模拟"从今天开始跟踪"）
            # 用户可通过 --reset --start-back N 指定回溯天数
            back_days = getattr(self, '_start_back_days', 60)
            start_idx = max(0, len(feat) - back_days)
            self.state.start_date = str(dates[start_idx])
            current_idx = start_idx
        else:
            # 增量运行：找到上次处理的日期之后的第一天
            last_d = pd.Timestamp(self.state.last_processed_date).date()
            current_idx = None
            for i in range(len(dates)):
                if pd.Timestamp(dates[i]).date() > last_d:
                    current_idx = i
                    break
            if current_idx is None:
                return self._today_snapshot(close[-1], atr[-1], probs[-1], str(dates[-1]))
        
        # 逐日推进
        days_processed = 0
        for i in range(current_idx, len(dates)):
            price = float(close[i])
            cur_atr = float(max(atr[i], 1.0))
            p_up = float(probs[i])
            date_str = str(dates[i])
            
            # 确定信号方向
            if p_up > self.buy_thr:
                signal_dir = 'long'
            elif p_up < self.sell_thr:
                signal_dir = 'short'
            else:
                signal_dir = 'neutral'
            
            # 1. 检查是否需要平仓
            pos = self.state.position
            if pos:
                exit_reason = None
                if pos['direction'] == 'long':
                    if price <= pos['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price >= pos['take_profit']:
                        exit_reason = 'take_profit'
                    elif signal_dir == 'short':
                        exit_reason = 'signal_flip'
                else:  # short
                    if price >= pos['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif price <= pos['take_profit']:
                        exit_reason = 'take_profit'
                    elif signal_dir == 'long':
                        exit_reason = 'signal_flip'
                
                if exit_reason:
                    # 平仓
                    if pos['direction'] == 'long':
                        pnl = (price - pos['entry_price']) * pos['quantity']
                    else:
                        pnl = (pos['entry_price'] - price) * pos['quantity']
                    
                    trade = {
                        'entry_date': pos['entry_date'],
                        'exit_date': date_str,
                        'direction': pos['direction'],
                        'entry_price': round(pos['entry_price'], 2),
                        'exit_price': round(price, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl / pos['entry_capital'] * 100, 2),
                        'exit_reason': exit_reason,
                        'signal_p': round(pos['entry_signal_p'], 3),
                    }
                    self.state.trades.append(trade)
                    self.state.capital += pnl
                    self.state.position = None
            
            # 2. 检查是否需要开仓
            if self.state.position is None and signal_dir in ('long', 'short'):
                confidence = abs(p_up - 0.5)
                p_win = p_up if signal_dir == 'long' else (1 - p_up)
                pos_pct = self._kelly_pct(p_win) * min(confidence / 0.15, 1.5)
                pos_pct = min(pos_pct, self.max_position)
                
                if pos_pct >= 0.02:
                    quantity = self.state.capital * pos_pct / price
                    
                    if signal_dir == 'long':
                        sl = price - self.stop_atr * cur_atr
                        tp = price + self.tp_atr * cur_atr
                    else:
                        sl = price + self.stop_atr * cur_atr
                        tp = price - self.tp_atr * cur_atr
                    
                    self.state.position = {
                        'direction': signal_dir,
                        'entry_price': round(price, 2),
                        'quantity': quantity,
                        'entry_date': date_str,
                        'entry_signal_p': round(p_up, 4),
                        'stop_loss': round(sl, 2),
                        'take_profit': round(tp, 2),
                        'entry_capital': self.state.capital,
                    }
            
            # 3. 记录当日权益
            if self.state.position:
                p = self.state.position
                if p['direction'] == 'long':
                    unrealized = (price - p['entry_price']) * p['quantity']
                else:
                    unrealized = (p['entry_price'] - price) * p['quantity']
                equity = self.state.capital + unrealized
            else:
                equity = self.state.capital
            
            self.state.equity_log.append({
                'date': date_str,
                'price': round(price, 2),
                'equity': round(equity, 2),
                'capital': round(self.state.capital, 2),
                'in_position': self.state.position is not None,
                'position_dir': self.state.position['direction'] if self.state.position else None,
                'p_up': round(p_up, 4),
                'atr': round(cur_atr, 2),
            })
            
            # 精简日志（保留最近500天，其余每5天采样）
            if len(self.state.equity_log) > 500:
                recent = self.state.equity_log[-500:]
                old = self.state.equity_log[:-500]
                sampled_old = old[::max(1, len(old) // 200)]
                self.state.equity_log = sampled_old + recent
            
            days_processed += 1
            self.state.last_processed_date = date_str
        
        # 保存状态
        self._save_state()
        
        # 返回今日快照
        return self._today_snapshot(
            float(close[-1]), float(atr[-1]), float(probs[-1]), str(dates[-1]),
            days_processed=days_processed
        )
    
    def _today_snapshot(self, price, atr_val, p_up, date_str, days_processed=0) -> Dict:
        """生成今日快照（供看板JSON）"""
        # 计算绩效
        total_return = (self.state.capital / self.state.initial_capital - 1) * 100
        
        # 胜率等
        trades = self.state.trades
        if trades:
            win_count = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = win_count / len(trades) * 100
            total_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            total_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
            profit_factor = total_wins / total_losses if total_losses > 0 else 999
        else:
            win_rate = profit_factor = 0
        
        # 最大回撤
        if self.state.equity_log:
            eq_vals = [e['equity'] for e in self.state.equity_log]
            peak = np.maximum.accumulate(eq_vals)
            dd = min((np.array(eq_vals) - peak) / peak * 100)
        else:
            dd = 0
        
        # 日收益率
        if len(self.state.equity_log) >= 30:
            eq_arr = np.array([e['equity'] for e in self.state.equity_log])
            daily_rets = np.diff(eq_arr) / eq_arr[:-1]
            daily_rets = daily_rets[np.isfinite(daily_rets)]
            if len(daily_rets) > 0 and daily_rets.std() > 0:
                sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std()
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # 当前持仓
        if self.state.position:
            p = self.state.position
            if p['direction'] == 'long':
                unrealized = (price - p['entry_price']) * p['quantity']
                unrealized_pct = (price / p['entry_price'] - 1) * 100
            else:
                unrealized = (p['entry_price'] - price) * p['quantity']
                unrealized_pct = (p['entry_price'] / price - 1) * 100
            position_info = {
                'direction': p['direction'],
                'entry_price': p['entry_price'],
                'entry_date': p['entry_date'],
                'stop_loss': p['stop_loss'],
                'take_profit': p['take_profit'],
                'unrealized_pnl': round(unrealized, 2),
                'unrealized_pnl_pct': round(unrealized_pct, 2),
            }
        else:
            position_info = None
        
        # 信号建议
        signal = {
            'date': date_str,
            'price': round(price, 2),
            'p_up': round(p_up, 4),
            'direction': 'BUY' if p_up > self.buy_thr else ('SELL' if p_up < self.sell_thr else 'HOLD'),
            'confidence': round(abs(p_up - 0.5), 4),
            'atr': round(atr_val, 2),
        }
        
        if position_info is None and signal['direction'] in ('BUY', 'SELL'):
            confidence = abs(p_up - 0.5)
            p_win = p_up if signal['direction'] == 'BUY' else (1 - p_up)
            pos_pct = min(self._kelly_pct(p_win) * min(confidence / 0.15, 1.5), self.max_position)
            is_long = signal['direction'] == 'BUY'
            signal['suggestion'] = {
                'action': f'开{"多" if is_long else "空"}仓',
                'position_pct': round(pos_pct * 100, 1),
                'stop_loss': round(price - self.stop_atr * atr_val if is_long else price + self.stop_atr * atr_val, 2),
                'take_profit': round(price + self.tp_atr * atr_val if is_long else price - self.tp_atr * atr_val, 2),
            }
        
        # 精简的交易记录（最近20笔）
        recent_trades = self.state.trades[-20:]
        
        # 权益曲线（采样）
        eq_log = self.state.equity_log
        if len(eq_log) > 300:
            step = len(eq_log) // 300
            eq_log = eq_log[::step] + [eq_log[-1]]
        
        return {
            'signal': signal,
            'position': position_info,
            'perf': {
                'initial_capital': self.state.initial_capital,
                'capital': round(self.state.capital, 2),
                'total_return_pct': round(total_return, 1),
                'win_rate_pct': round(win_rate, 1),
                'profit_factor': round(profit_factor, 2) if profit_factor < 999 else 999,
                'max_drawdown_pct': round(dd, 2),
                'sharpe': round(sharpe, 2),
                'n_trades': len(trades),
                'n_days': len(self.state.equity_log),
                'start_date': self.state.start_date,
            },
            'recent_trades': recent_trades,
            'equity_log': eq_log,
            'days_processed': days_processed,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


# ======================== 命令行入口 ========================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='黄金潮汐实盘模拟器')
    parser.add_argument('--reset', action='store_true', help='重置模拟状态（从今天重新开始）')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金（仅reset时生效）')
    parser.add_argument('--start-back', type=int, default=60, help='首次运行时回溯天数（默认60天）')
    parser.add_argument('--output', type=str, default='sim_live_result.json', help='结果JSON输出')
    args = parser.parse_args()
    
    if args.reset:
        if os.path.exists('sim_live_state.json'):
            os.remove('sim_live_state.json')
            print('已重置模拟状态')
        else:
            print('无需重置（无历史状态）')
    
    trader = LiveTrader(initial_capital=args.capital)
    trader._start_back_days = args.start_back  # 传回溯天数
    result = trader.run_daily()
    
    # 输出JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 打印简报
    s = result['signal']
    p = result['perf']
    print(f"\n{'='*50}")
    print(f"  实盘模拟简报 — {s['date']}")
    print(f"{'='*50}")
    print(f"  当前价: {s['price']}  |  模型P(涨): {s['p_up']:.4f}")
    print(f"  信号:   {s['direction']}  |  置信度: {s['confidence']:.4f}")
    print(f"  资金:   {p['initial_capital']:,.0f} → {p['capital']:,.0f}  ({p['total_return_pct']:+.1f}%)")
    print(f"  交易:   {p['n_trades']}笔  |  胜率: {p['win_rate_pct']:.1f}%")
    print(f"  起始:   {p['start_date']}  |  跟踪: {p['n_days']}天  |  今日推进: {result['days_processed']}天")
    
    pos = result['position']
    if pos:
        print(f"\n  持仓: {pos['direction']} @ {pos['entry_price']} (自{pos['entry_date']})")
        print(f"  浮动盈亏: {pos['unrealized_pnl_pct']:+.2f}%")
        print(f"  止损: {pos['stop_loss']}  |  止盈: {pos['take_profit']}")
    elif result['signal'].get('suggestion'):
        sug = result['signal']['suggestion']
        print(f"\n  建议: {sug['action']}  |  仓位: {sug['position_pct']}%")
        print(f"  止损: {sug['stop_loss']}  |  止盈: {sug['take_profit']}")
    
    print(f"\n  结果: {args.output}  |  状态: {trader.state_path}")
