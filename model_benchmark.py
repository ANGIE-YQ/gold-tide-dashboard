#!/usr/bin/env python3
"""
模型配置对比回测 — 标准化评测框架
对同一组数据跑多种配置，生成横向对比报告
"""

import numpy as np
import pandas as pd
import pickle, json, os, sys
from datetime import datetime
from collections import defaultdict

# ======================== 回测核心 ========================

class BacktestEngine:
    """Walk-Forward回测引擎"""
    
    def __init__(self, feat_path='gold_features_enhanced.csv', fwd=10,
                 train_window=1500, step_size=200,
                 raw_data_path='gold_AU0_daily.csv'):
        feat = pd.read_csv(feat_path, parse_dates=['Date'])
        self.close = feat['Close'].values.astype(float)
        self.dates = feat['Date'].values
        self.label_col = f'fwd{fwd}'
        self.y = (feat[self.label_col] > 0).astype(int).values
        self.valid = ~feat[self.label_col].isna().values
        self.feat = feat
        self.train_window = train_window
        self.step_size = step_size
        self.n = len(feat)
        # 加载原始 OHLCV 用于正确的 True Range 计算
        self.raw = pd.read_csv(raw_data_path, parse_dates=['Date'])
        self.raw = self.raw.sort_values('Date').reset_index(drop=True)
        self._compute_atr()
    
    def _compute_atr(self):
        """从原始OHLCV计算正确的True Range和ATR，按日期对齐到特征数据"""
        raw_high = self.raw['High'].values.astype(float)
        raw_low = self.raw['Low'].values.astype(float)
        raw_close = self.raw['Close'].values.astype(float)
        raw_dates = self.raw['Date'].values.astype('datetime64[D]')
        feat_dates = self.dates.astype('datetime64[D]')
        
        # 正确 True Range: max(H-L, |H-prevC|, |L-prevC|)
        n_raw = len(raw_close)
        tr = np.zeros(n_raw)
        for i in range(1, n_raw):
            tr[i] = max(
                raw_high[i] - raw_low[i],
                abs(raw_high[i] - raw_close[i-1]),
                abs(raw_low[i] - raw_close[i-1])
            )
        tr[0] = tr[1] if n_raw > 1 else 1.0
        
        raw_atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
        
        # 按日期对齐到特征数据
        date_to_atr = dict(zip(raw_dates, raw_atr))
        self.atr_arr = np.array([date_to_atr.get(d, raw_atr[-1]) for d in feat_dates])
        self.atr_arr = np.maximum(self.atr_arr, 1.0)  # 最低1.0防除零
    
    def run(self, config_name, feature_cols, thresholds,
            stop_atr=2.5, tp_atr=3.0, max_pos=0.25,
            external_probs=None):
        """
        执行一次回测
        threshold: (buy_thr, sell_thr) — 开仓条件 P>buy_thr → long, P<sell_thr → short
        external_probs: 可选 np.array(len(close)), 提供外部概率则跳过WF训练
        """
        from xgboost import XGBClassifier
        
        valid_idx = np.where(self.valid)[0]
        start_idx = valid_idx[self.train_window]
        if start_idx >= len(valid_idx):
            return None
        
        all_probs = np.full(self.n, 0.5)
        
        if external_probs is not None:
            # 外部概率模式：直接使用提供的概率，跳过WF训练
            all_probs = external_probs.copy()
        else:
            # Walk-Forward预测
            feat_cols_avail = [c for c in feature_cols if c in self.feat.columns]
            for test_start in range(start_idx, len(valid_idx), self.step_size):
                test_end = min(test_start + self.step_size, len(valid_idx))
                if test_start >= len(valid_idx):
                    break
                
                train_end_idx = valid_idx[test_start - 1]
                train_mask = (np.arange(self.n) >= valid_idx[0]) & (np.arange(self.n) <= train_end_idx) & self.valid
                X_train = self.feat.loc[train_mask, feat_cols_avail].values
                y_train = self.y[train_mask]
                
                if len(X_train) < 500:
                    continue
                
                test_start_orig = valid_idx[test_start]
                test_end_orig = valid_idx[min(test_end, len(valid_idx)-1)] if test_end > test_start else test_start_orig + 1
                test_mask = (np.arange(self.n) >= test_start_orig) & (np.arange(self.n) <= test_end_orig) & self.valid
                test_indices = np.where(test_mask)[0]
                if len(test_indices) == 0:
                    continue
                
                model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric='logloss', random_state=42)
                model.fit(X_train, y_train)
                probs = model.predict_proba(self.feat.loc[test_mask, feat_cols_avail].values)[:, 1]
                all_probs[test_indices] = probs
        
        # 交易模拟
        buy_thr, sell_thr = thresholds
        capital = 100000
        initial = capital
        position = None
        trades = []
        equity = [{'date': str(self.dates[valid_idx[start_idx]]), 'equity': capital, 'capital': capital}]
        
        # ATR (已从原始OHLCV预计算, 正确TR=max(H-L,|H-prevC|,|L-prevC|))
        atr_arr = self.atr_arr
        close = self.close
        
        for i in range(len(self.dates)):
            if not self.valid[i] or all_probs[i] == 0.5:
                continue
            p = all_probs[i]
            price = close[i]
            atr = max(atr_arr[i], 1.0)
            date_str = str(self.dates[i])
            
            # 信号
            if p > buy_thr:
                sig = 'long'
            elif p < sell_thr:
                sig = 'short'
            else:
                sig = 'neutral'
            
            # 平仓
            if position:
                exit_reason = None
                if position['dir'] == 'long':
                    if price <= position['sl']: exit_reason = 'stop'
                    elif price >= position['tp']: exit_reason = 'tp'
                    elif sig == 'short': exit_reason = 'flip'
                else:
                    if price >= position['sl']: exit_reason = 'stop'
                    elif price <= position['tp']: exit_reason = 'tp'
                    elif sig == 'long': exit_reason = 'flip'
                
                if exit_reason:
                    pnl = (price - position['entry']) * position['qty'] if position['dir'] == 'long' else (position['entry'] - price) * position['qty']
                    trades.append({'pnl': pnl, 'pnl_pct': pnl / capital * 100, 'reason': exit_reason,
                                   'entry_date': position['date'], 'exit_date': date_str, 'dir': position['dir']})
                    capital += pnl
                    position = None
            
            # 开仓
            if not position and sig != 'neutral':
                pos_pct = min(max_pos, 0.20 * (abs(p - 0.5) / 0.15))
                if pos_pct >= 0.03:
                    qty = capital * pos_pct / price
                    sl = price - stop_atr * atr if sig == 'long' else price + stop_atr * atr
                    tp = price + tp_atr * atr if sig == 'long' else price - tp_atr * atr
                    position = {'dir': sig, 'entry': price, 'qty': qty, 'sl': sl, 'tp': tp, 'date': date_str}
            
            # 权益
            if position:
                unreal = (price - position['entry']) * position['qty'] if position['dir'] == 'long' else (position['entry'] - price) * position['qty']
                eq = capital + unreal
            else:
                eq = capital
            equity.append({'date': date_str, 'equity': eq, 'capital': capital})
        
        # 强制平仓
        if position:
            pnl = (close[-1] - position['entry']) * position['qty'] if position['dir'] == 'long' else (position['entry'] - close[-1]) * position['qty']
            trades.append({'pnl': pnl, 'pnl_pct': pnl / capital * 100, 'reason': 'close', 'dir': position['dir'],
                           'entry_date': position['date'], 'exit_date': str(self.dates[-1])})
            capital += pnl
        
        # 绩效
        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
            total_wins = sum(t['pnl'] for t in wins)
            total_losses = abs(sum(t['pnl'] for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 999
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        eq_arr = np.array([e['equity'] for e in equity])
        peak = np.maximum.accumulate(eq_arr)
        mdd = float(np.min((eq_arr - peak) / peak * 100))
        total_ret = (capital / initial - 1) * 100
        years = max(len(equity) / 252, 0.5)
        cagr = (capital / initial) ** (1 / years) - 1
        
        daily_rets = np.diff(eq_arr) / eq_arr[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        sharpe = np.sqrt(252) * np.mean(daily_rets) / np.std(daily_rets) if len(daily_rets) > 0 and np.std(daily_rets) > 0 else 0
        
        return {
            'config': config_name,
            'thresholds': f'buy>{buy_thr} sell<{sell_thr}',
            'total_return_pct': round(total_ret, 1),
            'cagr_pct': round(cagr * 100, 1),
            'sharpe': round(sharpe, 2),
            'max_dd_pct': round(mdd, 1),
            'n_trades': len(trades),
            'win_rate_pct': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'final_capital': round(capital, 0),
        }


# ======================== 配置矩阵 ========================

def load_feature_sets():
    """定义要对比的特征集 (三档模型)"""
    with open('gold_tide_calibrated_model.pkl', 'rb') as f:
        pkg = pickle.load(f)
    
    all_features = pkg['features']
    dead = pkg.get('dead', [])
    base_features = [c for c in all_features if c not in dead]
    
    # 精简基线: 仅潮汐核心+拐点(18维, 最优压缩)
    slim_features = [c for c in base_features
                     if c.startswith(('cur_', 'prev_', 'lp_'))]
    
    # ±En·ʋ 特征 (动态检测CSV中存在的)
    feat_check = pd.read_csv('gold_features_enhanced.csv', nrows=1)
    en_v_features = [c for c in feat_check.columns if c.startswith('en_v_')]
    
    return {
        '原始(31维)': base_features,
        '精简(18维)': slim_features,
    } | ({
        '原始+±En·v': base_features + en_v_features,
        '精简+±En·v': slim_features + en_v_features,
        '纯±En·v(24维)': en_v_features,
    } if en_v_features else {})


def run_comparison():
    """运行所有配置的回测对比"""
    print('=' * 60)
    print('  模型配置对比回测')
    print(f'  时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)
    
    engine = BacktestEngine()
    feature_sets = load_feature_sets()
    
    # 阈值: 只保留保守和标准两档 (宽松/激进边际价值低)
    threshold_pairs = [
        (0.65, 0.35, '保守'),
        (0.60, 0.40, '标准'),
    ]
    
    results = []
    for feat_name, feat_cols in feature_sets.items():
        for buy_t, sell_t, t_name in threshold_pairs:
            config = f'{feat_name} | {t_name}阈值'
            print(f'  测试: {config}...', end=' ')
            r = engine.run(config, feat_cols, (buy_t, sell_t))
            if r:
                print(f'收益{r["total_return_pct"]:+.1f}% 胜率{r["win_rate_pct"]:.0f}% 夏普{r["sharpe"]:.2f}')
                results.append(r)
            else:
                print('跳过')
    
    # 排名
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # 输出报告
    print(f'\n{"=" * 60}')
    print(f'  排名（按夏普比率）')
    print(f'{"=" * 60}')
    print(f'{"名次":<4} {"配置":<35} {"收益":>8} {"夏普":>6} {"回撤":>6} {"胜率":>6} {"交易":>5} {"盈亏比":>6}')
    print('-' * 80)
    
    for i, r in enumerate(results):
        rank = f'#{i+1}'
        print(f'{rank:<4} {r["config"]:<35} {r["total_return_pct"]:>+7.1f}% {r["sharpe"]:>6.2f} {r["max_dd_pct"]:>5.1f}% {r["win_rate_pct"]:>5.0f}% {r["n_trades"]:>4}笔 {r["profit_factor"]:>6.1f}')
    
    # 最优 vs 最差 vs 当前
    best = results[0]
    worst = results[-1]
    current = [r for r in results if '当前模型(31维)' in r['config'] and '标准' in r['config']]
    
    print(f'\n{"=" * 60}')
    print(f'  结论')
    print(f'{"=" * 60}')
    print(f'  最优配置: {best["config"]}')
    print(f'    收益={best["total_return_pct"]:+.1f}% 夏普={best["sharpe"]:.2f} 回撤={best["max_dd_pct"]:.1f}% 胜率={best["win_rate_pct"]:.0f}%')
    
    if current:
        cur = current[0]
        print(f'\n  当前配置: {cur["config"]}')
        print(f'    收益={cur["total_return_pct"]:+.1f}% 夏普={cur["sharpe"]:.2f} 回撤={cur["max_dd_pct"]:.1f}% 胜率={cur["win_rate_pct"]:.0f}%')
        
        improvement = (best['sharpe'] - cur['sharpe']) / max(abs(cur['sharpe']), 0.01) * 100
        print(f'\n  → 切换到最优配置可提升夏普 {improvement:+.0f}%')
    
    # 保存结果
    with open('model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n  详细结果: model_comparison_results.json')


if __name__ == '__main__':
    run_comparison()
