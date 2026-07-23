#!/usr/bin/env python3
"""
黄金潮汐模型 — 交易模拟器
支持：Walk-Forward回测 + 实时模拟 + 绩效分析
"""

import numpy as np
import pandas as pd
import pickle, json, os, sys, time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ======================== 数据模型 ========================

@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: str
    exit_date: str
    direction: str          # 'long' / 'short'
    entry_price: float
    exit_price: float
    quantity: float
    entry_capital: float    # 入场时总资金
    pnl: float              # 盈亏金额
    pnl_pct: float          # 盈亏百分比(vs 总资金)
    exit_reason: str        # 'stop_loss' / 'take_profit' / 'signal_flip' / 'close_all'
    signal_p: float         # 入场时的模型P值
    confidence: float       # 置信度

@dataclass
class Position:
    """当前持仓"""
    direction: str
    entry_price: float
    quantity: float
    entry_date: str
    entry_signal_p: float
    stop_loss: float
    take_profit: float

@dataclass 
class SimState:
    """模拟器状态"""
    capital: float
    initial_capital: float
    position: Optional[Position] = None
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

# ======================== 核心引擎 ========================

class TideTrader:
    """
    潮汐交易模拟器
    
    策略规则：
    - 信号: 校准模型 P(涨) > 0.60 → long, P(涨) < 0.40 → short
    - 仓位: Kelly公式 × 置信度, 上限25%
    - 止损: 2.5 × ATR
    - 止盈: 3.0 × ATR
    - 每日检查一次信号和止损止盈
    """
    
    def __init__(self, initial_capital=100000, max_position=0.25,
                 stop_atr_mult=2.5, tp_atr_mult=3.0,
                 buy_threshold=0.60, sell_threshold=0.40):
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.stop_atr_mult = stop_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.state = SimState(capital=initial_capital, initial_capital=initial_capital)
        
    def kelly_position(self, p_win: float, win_loss_ratio=1.5) -> float:
        """Kelly公式计算仓位比例"""
        p_lose = 1 - p_win
        kelly = (p_win * win_loss_ratio - p_lose) / win_loss_ratio
        return max(0, min(kelly * 0.5, self.max_position))  # 半凯利,上限25%
    
    def check_exit(self, price: float, atr: float, signal_direction: str) -> Optional[str]:
        """检查是否需要平仓"""
        pos = self.state.position
        if pos is None:
            return None
        
        if pos.direction == 'long':
            if price <= pos.stop_loss:
                return 'stop_loss'
            if price >= pos.take_profit:
                return 'take_profit'
            if signal_direction == 'short':  # 信号反转
                return 'signal_flip'
        else:  # short
            if price >= pos.stop_loss:
                return 'stop_loss'
            if price <= pos.take_profit:
                return 'take_profit'
            if signal_direction == 'long':
                return 'signal_flip'
        return None
    
    def close_position(self, price: float, date: str, reason: str, capital: float):
        """平仓并记录交易"""
        pos = self.state.position
        if pos is None:
            return
        
        if pos.direction == 'long':
            pnl = (price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity
        
        pnl_pct = pnl / capital
        
        trade = Trade(
            entry_date=pos.entry_date,
            exit_date=date,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_capital=capital,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            signal_p=pos.entry_signal_p,
            confidence=abs(pos.entry_signal_p - 0.5)
        )
        
        self.state.trades.append(trade)
        self.state.position = None
        
    def open_position(self, direction: str, price: float, atr: float, 
                      signal_p: float, date: str, capital: float):
        """开仓"""
        if self.state.position is not None:
            return  # 已有仓位
        
        confidence = abs(signal_p - 0.5)
        kelly_pct = self.kelly_position(signal_p if direction == 'long' else (1 - signal_p))
        position_pct = kelly_pct * (confidence / 0.15)  # 置信度调整
        position_pct = min(position_pct, self.max_position)
        
        if position_pct < 0.02:  # 最低2%
            return
        
        quantity = capital * position_pct / price
        
        if direction == 'long':
            sl = price - self.stop_atr_mult * atr
            tp = price + self.tp_atr_mult * atr
        else:
            sl = price + self.stop_atr_mult * atr
            tp = price - self.tp_atr_mult * atr
        
        self.state.position = Position(
            direction=direction,
            entry_price=price,
            quantity=quantity,
            entry_date=date,
            entry_signal_p=signal_p,
            stop_loss=sl,
            take_profit=tp
        )
    
    def step(self, price: float, atr: float, signal_p: float, date: str) -> Dict:
        """
        每日执行一步
        返回: 当天的状态快照
        """
        capital = self.state.capital
        
        # 确定信号方向
        if signal_p > self.buy_threshold:
            signal_dir = 'long'
        elif signal_p < self.sell_threshold:
            signal_dir = 'short'
        else:
            signal_dir = 'neutral'
        
        # 1. 检查是否需要平仓
        exit_reason = self.check_exit(price, atr, signal_dir)
        if exit_reason:
            self.close_position(price, date, exit_reason, capital)
            # 平仓后更新资金
            if self.state.position is None and self.state.trades:
                last_trade = self.state.trades[-1]
                self.state.capital += last_trade.pnl
                capital = self.state.capital
        
        # 2. 检查是否需要开仓（只在无仓位时）
        if self.state.position is None:
            if signal_dir in ('long', 'short'):
                self.open_position(signal_dir, price, atr, signal_p, date, capital)
        
        # 3. 记录权益曲线
        if self.state.position:
            if self.state.position.direction == 'long':
                unrealized = (price - self.state.position.entry_price) * self.state.position.quantity
            else:
                unrealized = (self.state.position.entry_price - price) * self.state.position.quantity
            equity = capital + unrealized
        else:
            equity = self.state.capital
        
        self.state.equity_curve.append({
            'date': date,
            'price': price,
            'equity': equity,
            'capital': self.state.capital,
            'in_position': self.state.position is not None,
            'position_dir': self.state.position.direction if self.state.position else None,
            'signal_p': signal_p,
            'atr': atr
        })
        
        return self.state.equity_curve[-1]
    
    def force_close_all(self, price: float, date: str):
        """强制平仓（回测结束或止损）"""
        if self.state.position:
            capital = self.state.capital
            self.close_position(price, date, 'close_all', capital)
            if self.state.position is None:
                self.state.capital += self.state.trades[-1].pnl
    
    def get_performance(self) -> Dict:
        """计算绩效指标"""
        if not self.state.equity_curve:
            return {'error': 'no data'}
        
        equity = pd.DataFrame(self.state.equity_curve)
        n_days = len(equity)
        
        # 总收益
        total_return = (self.state.capital / self.initial_capital - 1) * 100
        
        # 年化
        years = n_days / 252
        cagr = (self.state.capital / self.initial_capital) ** (1 / max(years, 0.01)) - 1
        
        # 日收益率
        eq_values = equity['equity'].values
        daily_rets = np.diff(eq_values) / eq_values[:-1]
        daily_rets = daily_rets[~np.isnan(daily_rets)]
        
        # 夏普
        if len(daily_rets) > 0 and daily_rets.std() > 0:
            sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std()
        else:
            sharpe = 0
        
        # 最大回撤
        peak = np.maximum.accumulate(eq_values)
        drawdowns = (eq_values - peak) / peak * 100
        max_dd = drawdowns.min()
        
        # 交易统计
        trades = self.state.trades
        if trades:
            win_trades = [t for t in trades if t.pnl > 0]
            loss_trades = [t for t in trades if t.pnl <= 0]
            win_rate = len(win_trades) / len(trades) * 100
            
            avg_win = np.mean([t.pnl_pct for t in win_trades]) * 100 if win_trades else 0
            avg_loss = np.mean([t.pnl_pct for t in loss_trades]) * 100 if loss_trades else 0
            
            total_wins = sum(t.pnl for t in win_trades)
            total_losses = abs(sum(t.pnl for t in loss_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # 按退出原因分组
            by_reason = {}
            for t in trades:
                r = t.exit_reason
                by_reason.setdefault(r, []).append(t)
            reason_stats = {r: {
                'count': len(ts),
                'win_rate': sum(1 for t in ts if t.pnl > 0) / len(ts) * 100,
                'avg_pnl_pct': np.mean([t.pnl_pct for t in ts]) * 100
            } for r, ts in by_reason.items()}
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            reason_stats = {}
        
        # 持仓统计
        in_pos = equity['in_position'].sum()
        pos_pct = in_pos / n_days * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(self.state.capital, 2),
            'total_return_pct': round(total_return, 2),
            'cagr_pct': round(cagr * 100, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'n_trades': len(trades),
            'win_rate_pct': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
            'position_ratio_pct': round(pos_pct, 1),
            'reason_stats': reason_stats,
            'n_days': n_days,
        }


# ======================== 回测模式 ========================

def run_backtest(feat_path='gold_features_enhanced.csv',
                 model_path='gold_tide_calibrated_model.pkl',
                 initial_capital=100000, fwd=10) -> Dict:
    """
    Walk-Forward 回测
    用扩展窗口训练 → 预测下一段 → 交易模拟
    """
    print(f'[回测] 加载数据...')
    
    # 加载特征
    feat = pd.read_csv(feat_path, parse_dates=['Date'])
    
    # 加载校准模型获取特征列表
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    feature_cols = model_pkg['features']
    dead_cols = model_pkg.get('dead', [])
    feat_cols = [c for c in feature_cols if c in feat.columns and c not in dead_cols]
    
    # 标签
    label_col = f'fwd{fwd}'
    y = (feat[label_col] > 0).astype(int).values
    valid_mask = ~feat[label_col].isna().values
    
    # ATR计算（用原始Close）
    close = feat['Close'].values.astype(float)
    dates = feat['Date'].dt.strftime('%Y-%m-%d').values
    
    # 简易ATR
    high = feat.get('High', close).values.astype(float) if 'High' in feat.columns else close
    low = feat.get('Low', close).values.astype(float) if 'Low' in feat.columns else close
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    
    # 初始化
    trader = TideTrader(initial_capital=initial_capital)
    
    # Walk-Forward 参数
    train_window = 1500  # 初始训练窗口
    step_size = 200       # 每步新增数据量
    
    print(f'[回测] Walk-Forward: train={train_window}, step={step_size}')
    print(f'[回测] 特征数: {len(feat_cols)}, 有效样本: {valid_mask.sum()}')
    
    # 获取训练/预测
    valid_idx = np.where(valid_mask)[0]
    start_idx = valid_idx[train_window]  # 从第train_window个有效样本后开始
    
    all_signals = np.full(len(feat), 0.5)  # 存储所有信号P值
    
    # Walk-Forward 循环
    wf_start = valid_idx[0]
    n_steps = 0
    for test_start in range(start_idx, len(valid_idx), step_size):
        test_end = min(test_start + step_size, len(valid_idx))
        if test_start >= len(valid_idx):
            break
        train_end_idx = valid_idx[test_start - 1]
        
        # 训练集
        train_mask = (np.arange(len(feat)) >= wf_start) & (np.arange(len(feat)) <= train_end_idx) & valid_mask
        X_train = feat.loc[train_mask, feat_cols].values
        y_train = y[train_mask]
        
        if len(X_train) < 500:
            continue
        
        # 测试集
        test_start_orig = valid_idx[test_start]
        test_end_orig = valid_idx[min(test_end, len(valid_idx)-1)] if test_end > test_start else test_start_orig + 1
        test_mask = (np.arange(len(feat)) >= test_start_orig) & (np.arange(len(feat)) < test_end_orig) & valid_mask
        test_indices = np.where(test_mask)[0]
        
        if len(test_indices) == 0:
            continue
        
        # 训练模型
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42
        )
        model.fit(X_train, y_train)
        
        # 预测
        X_test = feat.loc[test_mask, feat_cols].values
        probs = model.predict_proba(X_test)[:, 1]
        all_signals[test_indices] = probs
        
        n_steps += 1
        
    print(f'[回测] 完成 {n_steps} 步预测, 覆盖 {int((all_signals != 0.5).sum())} 个交易日')
    
    # 运行交易模拟
    print(f'[回测] 运行交易模拟...')
    for i in range(len(feat)):
        if not valid_mask[i] or all_signals[i] == 0.5:
            continue  # 跳过无效或无信号的日子
        
        signal_p = all_signals[i]
        price = close[i]
        cur_atr = max(atr[i], 1.0)
        date_str = dates[i]
        
        trader.step(price, cur_atr, signal_p, date_str)
    
    # 强制平仓
    final_price = close[-1]
    final_date = dates[-1]
    trader.force_close_all(final_price, final_date)
    
    # 绩效
    perf = trader.get_performance()
    
    # 保存结果
    result = {
        'perf': perf,
        'equity_curve': trader.state.equity_curve,
        'trades': [{
            'entry': t.entry_date,
            'exit': t.exit_date,
            'dir': t.direction,
            'entry_price': round(t.entry_price, 2),
            'exit_price': round(t.exit_price, 2),
            'pnl': round(t.pnl, 2),
            'pnl_pct': round(t.pnl_pct * 100, 2),
            'reason': t.exit_reason,
            'signal_p': round(t.signal_p, 3),
        } for t in trader.state.trades],
        'params': {
            'initial_capital': initial_capital,
            'stop_atr': trader.stop_atr_mult,
            'tp_atr': trader.tp_atr_mult,
            'buy_threshold': trader.buy_threshold,
            'sell_threshold': trader.sell_threshold,
            'max_position': trader.max_position,
        }
    }
    
    return result


# ======================== 实时模拟模式 ========================

def run_live_simulation(feat_path='gold_features_enhanced.csv',
                        model_path='gold_tide_calibrated_model.pkl',
                        state_path='sim_state.json') -> Dict:
    """
    实时模拟：基于最新模型预测给出交易建议
    同时读取/保存模拟状态
    """
    # 加载模型
    with open(model_path, 'rb') as f:
        model_pkg = pickle.load(f)
    
    calibrated = model_pkg['model']
    feature_cols = model_pkg['features']
    dead_cols = model_pkg.get('dead', [])
    
    # 加载最新特征
    feat = pd.read_csv(feat_path, parse_dates=['Date'])
    feat_cols = [c for c in feature_cols if c in feat.columns and c not in dead_cols]
    
    last_row = feat.iloc[-1]
    X = feat[feat_cols].iloc[-1:].values
    
    # 预测
    p_up = float(calibrated.predict_proba(X)[0, 1])
    
    # 价格和ATR
    close = feat['Close'].values.astype(float)
    price = close[-1]
    
    # 简易ATR
    high = feat.get('High', close).values.astype(float) if 'High' in feat.columns else close
    low = feat.get('Low', close).values.astype(float) if 'Low' in feat.columns else close
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    atr_val = float(pd.Series(tr).rolling(20, min_periods=5).mean().iloc[-1])
    
    date_str = str(last_row['Date'].date())
    
    # 读取历史模拟状态
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            sim_state = json.load(f)
    else:
        sim_state = {
            'capital': 100000,
            'initial_capital': 100000,
            'position': None,
            'trade_count': 0,
            'total_pnl': 0,
        }
    
    # 生成交易建议
    trader = TideTrader(initial_capital=sim_state['initial_capital'])
    trader.state.capital = sim_state['capital']
    
    # 恢复持仓
    if sim_state.get('position'):
        pos_data = sim_state['position']
        trader.state.position = Position(
            direction=pos_data['direction'],
            entry_price=pos_data['entry_price'],
            quantity=pos_data['quantity'],
            entry_date=pos_data['entry_date'],
            entry_signal_p=pos_data['entry_signal_p'],
            stop_loss=pos_data['stop_loss'],
            take_profit=pos_data['take_profit']
        )
    
    # 执行一步
    step_result = trader.step(price, atr_val, p_up, date_str)
    
    # 更新模拟状态
    new_state = {
        'capital': trader.state.capital,
        'initial_capital': sim_state['initial_capital'],
        'position': None,
        'trade_count': sim_state['trade_count'] + (1 if trader.state.trades and 
                      trader.state.trades[-1].exit_date == date_str else 0),
        'total_pnl': trader.state.capital - sim_state['initial_capital'],
    }
    
    if trader.state.position:
        new_state['position'] = {
            'direction': trader.state.position.direction,
            'entry_price': trader.state.position.entry_price,
            'quantity': trader.state.position.quantity,
            'entry_date': trader.state.position.entry_date,
            'entry_signal_p': trader.state.position.entry_signal_p,
            'stop_loss': trader.state.position.stop_loss,
            'take_profit': trader.state.position.take_profit,
        }
    
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(new_state, f, ensure_ascii=False, indent=2)
    
    # 生成建议
    signal = {
        'date': date_str,
        'price': round(price, 2),
        'p_up': round(p_up, 4),
        'direction': 'BUY' if p_up > 0.60 else ('SELL' if p_up < 0.40 else 'HOLD'),
        'confidence': round(abs(p_up - 0.5), 4),
        'atr': round(atr_val, 2),
    }
    
    # 开仓建议
    if trader.state.position:
        pos = trader.state.position
        signal['position'] = {
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'entry_date': pos.entry_date,
            'stop_loss': round(pos.stop_loss, 2),
            'take_profit': round(pos.take_profit, 2),
            'unrealized_pnl_pct': round(
                (price - pos.entry_price) / pos.entry_price * 100 if pos.direction == 'long'
                else (pos.entry_price - price) / pos.entry_price * 100, 2
            ),
        }
    else:
        if signal['direction'] in ('BUY', 'SELL'):
            kelly = trader.kelly_position(p_up if signal['direction'] == 'BUY' else (1 - p_up))
            position_pct = min(kelly * (signal['confidence'] / 0.15), trader.max_position)
            
            if signal['direction'] == 'BUY':
                sl = price - 2.5 * atr_val
                tp = price + 3.0 * atr_val
            else:
                sl = price + 2.5 * atr_val
                tp = price - 3.0 * atr_val
            
            signal['suggestion'] = {
                'action': f'开{signal["direction"]}仓',
                'position_pct': round(position_pct * 100, 1),
                'entry': round(price, 2),
                'stop_loss': round(sl, 2),
                'take_profit': round(tp, 2),
            }
    
    signal['state'] = new_state
    
    return signal


# ======================== 命令行入口 ========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='黄金潮汐交易模拟器')
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                       help='回测(backtest) 或 实时模拟(live)')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--output', type=str, default='sim_result.json', help='结果输出文件')
    parser.add_argument('--fwd', type=int, default=10, help='预测目标(日)')
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        print('=' * 60)
        print('  黄金潮汐模型 — Walk-Forward 回测')
        print('=' * 60)
        result = run_backtest(initial_capital=args.capital, fwd=args.fwd)
        
        perf = result['perf']
        print(f'\n{"=" * 60}')
        print(f'  绩效报告')
        print(f'{"=" * 60}')
        print(f'  初始资金:    {perf["initial_capital"]:,.0f}')
        print(f'  最终资金:    {perf["final_capital"]:,.0f}')
        print(f'  总收益率:    {perf["total_return_pct"]:+.1f}%')
        print(f'  年化收益:    {perf["cagr_pct"]:+.1f}%')
        print(f'  夏普比率:    {perf["sharpe"]:.2f}')
        print(f'  最大回撤:    {perf["max_drawdown_pct"]:.1f}%')
        print(f'  交易次数:    {perf["n_trades"]}')
        print(f'  胜率:        {perf["win_rate_pct"]:.1f}%')
        print(f'  平均盈利:    {perf["avg_win_pct"]:+.2f}%')
        print(f'  平均亏损:    {perf["avg_loss_pct"]:+.2f}%')
        print(f'  盈亏比:      {perf["profit_factor"]:.2f}')
        print(f'  持仓比例:    {perf["position_ratio_pct"]:.1f}%')
        
        if perf.get('reason_stats'):
            print(f'\n  按退出原因:')
            for r, s in perf['reason_stats'].items():
                print(f'    {r}: {s["count"]}笔, 胜率{s["win_rate"]:.0f}%, 平均{s["avg_pnl_pct"]:+.2f}%')
        
        # 保存JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\n  结果已保存: {args.output}')
    
    elif args.mode == 'live':
        print('[实时模拟] 加载模型...')
        signal = run_live_simulation()
        print(f'\n  === 当前信号 ===')
        print(f'  日期: {signal["date"]}')
        print(f'  价格: {signal["price"]}')
        print(f'  P(涨): {signal["p_up"]:.4f}')
        print(f'  方向: {signal["direction"]}')
        print(f'  置信: {signal["confidence"]:.4f}')
        if signal.get('position'):
            p = signal['position']
            print(f'\n  当前持仓: {p["direction"]} @ {p["entry_price"]}')
            print(f'  浮动盈亏: {p["unrealized_pnl_pct"]:+.2f}%')
            print(f'  止损: {p["stop_loss"]}')
            print(f'  止盈: {p["take_profit"]}')
        elif signal.get('suggestion'):
            s = signal['suggestion']
            print(f'\n  建议: {s["action"]}')
            print(f'  仓位: {s["position_pct"]}%')
            print(f'  止损: {s["stop_loss"]}')
            print(f'  止盈: {s["take_profit"]}')
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(signal, f, ensure_ascii=False, indent=2)
        print(f'\n  结果已保存: {args.output}')


if __name__ == '__main__':
    main()
