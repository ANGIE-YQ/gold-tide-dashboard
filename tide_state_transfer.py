#!/usr/bin/env python3
"""
±En·ʋ = En+1 — 潮汐状态转移预测框架

框架定义:
  ±   状态参数: 当前方向的优势度。正(+)=上涨占优, 负(-)=下跌占优。
              状态持续时间有限, 同状态内潮汐才能比较, 不同层级间独立。
  En  能量序列: 同状态内每个小潮汐的能量(高度×时间), 涨跌分开计算构成数集。
  ʋ   趋变系数: En数列的增益/衰减率。Enₙ₊₁ / Enₙ, 仅在同状态内有效。
  En+1 预测: 前一段能量×ʋ → 下一段能量。符合预期=状态持续, 不符=状态改变。

核心信号: 状态参数翻转时(优势度→0), 即为建仓时机。
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Tuple


def compute_state_advantage(up_energies: List[float], down_energies: List[float]) -> float:
    """
    计算状态优势度 ±
    +1 = 完全上涨占优, -1 = 完全下跌占优, 0 = 平衡
    """
    if not up_energies and not down_energies:
        return 0.0
    if not up_energies:
        return -1.0
    if not down_energies:
        return 1.0
    
    up_total = np.sum(up_energies)
    down_total = np.sum(down_energies)
    total = up_total + down_total
    
    if total == 0:
        return 0.0
    return (up_total - down_total) / total  # [-1, 1]


def compute_trend_coefficient(energies: List[float]) -> Tuple[float, str]:
    """
    计算趋变系数 ʋ
    对同方向能量序列, 计算相邻比值的几何平均
    返回: (ʋ, 趋势描述)
    """
    if len(energies) < 2:
        return 1.0, '数据不足'
    
    ratios = []
    for i in range(1, len(energies)):
        if energies[i-1] > 0:
            ratios.append(energies[i] / energies[i-1])
    
    if not ratios:
        return 1.0, '无有效比值'
    
    ʋ = float(np.exp(np.mean(np.log(ratios))))  # 几何平均
    
    if ʋ > 1.15:
        trend = '加速增强'
    elif ʋ > 1.03:
        trend = '温和增强'
    elif ʋ > 0.97:
        trend = '趋稳'
    elif ʋ > 0.85:
        trend = '温和衰减'
    else:
        trend = '加速衰减'
    
    return ʋ, trend


def predict_next_energy(energies: List[float], ʋ: float) -> float:
    """En+1 = En_last × ʋ"""
    if not energies:
        return 0
    return energies[-1] * ʋ


def analyze_tide_state(df, small_legs, window=90) -> Dict:
    """
    完整 ±En·ʋ 框架分析
    
    返回: {
        state_advantage: 当前优势度 [-1, 1]
        state_direction: 'up_dominant' / 'down_dominant' / 'balanced'
        up_sequence: 上涨能量序列
        down_sequence: 下跌能量序列
        up_v: 上涨趋变系数
        down_v: 下跌趋变系数
        up_trend: 上涨趋势描述
        down_trend: 下跌趋势描述
        predicted_up_next: 预计下一上涨能量
        predicted_down_next: 预计下一下跌能量
        state_continuation: 状态是否可能持续
        reversal_signal: 反转信号 (advantage接近0时触发)
        signal_strength: 信号强度 0-1
    }
    """
    close = df['Close'].values.astype(float)
    dates = df['Date'].astype(str).values
    n = len(df)
    
    # 确保energy字段
    for s in small_legs:
        if 'energy' not in s:
            s['energy'] = s['height'] * max(s['duration_bars'], 1)
    
    # 取最近window天内的已完成小潮汐
    recent_small = [s for s in small_legs if s['end_idx'] <= n-1 and s['end_idx'] >= n-1-window]
    
    if len(recent_small) < 4:
        return {'state_advantage': 0.0, 'status': 'data_insufficient'}
    
    # 分离涨跌能量序列
    up_seq = [s['energy'] for s in recent_small if s['direction'] == 'up']
    down_seq = [s['energy'] for s in recent_small if s['direction'] == 'down']
    
    # 1. 状态参数 ±
    advantage = compute_state_advantage(up_seq, down_seq)
    
    if advantage > 0.15:
        state_dir = 'up_dominant'
    elif advantage < -0.15:
        state_dir = 'down_dominant'
    else:
        state_dir = 'balanced'
    
    # 2. 趋变系数 ʋ
    up_v, up_trend = compute_trend_coefficient(up_seq)
    down_v, down_trend = compute_trend_coefficient(down_seq)
    
    # 3. 预测 En+1
    pred_up = predict_next_energy(up_seq, up_v)
    pred_down = predict_next_energy(down_seq, down_v)
    
    # 4. 状态持续判断
    # 如果上涨趋变系数>1且下跌趋变系数<1 → 上涨状态持续
    # 如果上涨趋变系数<1且下跌趋变系数>1 → 可能反转
    # 如果两边同时趋弱或趋强 → 优势度缩小, 可能反转
    
    if state_dir == 'up_dominant':
        if up_v < 0.95 and down_v > 1.05:
            continuation = 'warning'  # 上涨减弱+下跌增强→可能反转
        elif up_v > 1.03:
            continuation = 'likely'   # 上涨持续增强
        else:
            continuation = 'uncertain'
    elif state_dir == 'down_dominant':
        if down_v < 0.95 and up_v > 1.05:
            continuation = 'warning'  # 下跌减弱+上涨增强→可能反转
        elif down_v > 1.03:
            continuation = 'likely'   # 下跌持续增强
        else:
            continuation = 'uncertain'
    else:
        # 已平衡→关注哪边先打破平衡
        if up_v > down_v and up_v > 1.03:
            continuation = 'up_breakout'
        elif down_v > up_v and down_v > 1.03:
            continuation = 'down_breakout'
        else:
            continuation = 'neutral'
    
    # 5. 反转信号 (优势度→0)
    abs_adv = abs(advantage)
    if abs_adv < 0.05:
        reversal_signal = True
        signal_strength = 1.0
        signal_detail = '⚡⚡ 状态参数归零: 上涨和下跌动能完全平衡, 强烈反转信号'
    elif abs_adv < 0.10:
        reversal_signal = True
        signal_strength = 1.0 - (abs_adv - 0.05) / 0.05
        signal_detail = '⚡ 状态参数趋零: 优势度快速衰减, 即将平衡, 关注建仓'
    elif abs_adv < 0.20 and continuation in ('warning', 'uncertain'):
        reversal_signal = 'approaching'
        signal_strength = 0.5
        signal_detail = '状态参数逼近零区, 反转概率上升'
    else:
        reversal_signal = False
        signal_strength = 0.0
        signal_detail = ''
    
    return {
        'date': str(dates[-1]),
        'price': round(float(close[-1]), 2),
        'state_advantage': round(float(advantage), 4),
        'state_advantage_pct': round(float(advantage) * 100, 1),
        'state_direction': state_dir,
        'up_sequence': [round(x, 0) for x in up_seq[-5:]],
        'down_sequence': [round(x, 0) for x in down_seq[-5:]],
        'n_up': len(up_seq),
        'n_down': len(down_seq),
        'up_energy_total': round(float(np.sum(up_seq)), 0),
        'down_energy_total': round(float(np.sum(down_seq)), 0),
        'up_v': round(float(up_v), 4),
        'down_v': round(float(down_v), 4),
        'up_trend': up_trend,
        'down_trend': down_trend,
        'predicted_up_next': round(float(pred_up), 0),
        'predicted_down_next': round(float(pred_down), 0),
        'continuation': continuation,
        'reversal_signal': reversal_signal,
        'signal_strength': round(float(signal_strength), 4),
        'signal_detail': signal_detail,
        'n_recent_tides': len(recent_small),
    }


def get_signal_interpretation(result: Dict) -> str:
    """生成人类可读的信号解读"""
    if result.get('status') == 'data_insufficient':
        return '数据不足, 无法判断'
    
    adv = result['state_advantage']
    adv_pct = result['state_advantage_pct']
    state = result['state_direction']
    up_v = result['up_v']
    down_v = result['down_v']
    cont = result['continuation']
    sig = result['reversal_signal']
    detail = result['signal_detail']
    
    lines = []
    lines.append(f"状态参数 ± = {adv:+.3f} ({adv_pct:+.1f}%优势度)")
    
    if state == 'up_dominant':
        lines.append(f"当前: 上涨占优 | 上涨ʋ={up_v:.3f}({result['up_trend']}) | 下跌ʋ={down_v:.3f}({result['down_trend']})")
    elif state == 'down_dominant':
        lines.append(f"当前: 下跌占优 | 下跌ʋ={down_v:.3f}({result['down_trend']}) | 上涨ʋ={up_v:.3f}({result['up_trend']})")
    else:
        lines.append(f"当前: 多空平衡 | 上涨ʋ={up_v:.3f} | 下跌ʋ={down_v:.3f}")
    
    # 预测
    pred_up = result['predicted_up_next']
    pred_down = result['predicted_down_next']
    last_up = result['up_sequence'][-1] if result['up_sequence'] else 0
    last_down = result['down_sequence'][-1] if result['down_sequence'] else 0
    lines.append(f"预测: 下一上涨En+1≈{pred_up:.0f} (现{last_up:.0f}×{up_v:.3f}) | 下一下跌En+1≈{pred_down:.0f} (现{last_down:.0f}×{down_v:.3f})")
    
    if cont == 'warning':
        lines.append(f"⚠ 状态持续警告: 优势方向动能衰减, 对手方增强, 反转风险上升")
    elif cont == 'likely':
        lines.append(f"✓ 状态可能持续: 优势方向动能仍在增强")
    elif cont in ('up_breakout', 'down_breakout'):
        lines.append(f"→ 平衡态可能被打破: {cont}")
    
    if sig:
        lines.append(detail)
    
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    from gold_tide_engine import load_data, compute_atr, detect_tides
    from gold_tide_score import score_all_tides
    
    df = load_data('./gold_AU0_daily.csv')
    atr = compute_atr(df, 20)
    big, small, _, _ = detect_tides(df, 3.0, 1.0, atr)
    score_all_tides(df, big, small, atr)
    
    result = analyze_tide_state(df, small)
    interpretation = get_signal_interpretation(result)
    
    print(f"\n{'='*60}")
    print(f"  ±En·ʋ 潮汐状态分析 — {result['date']}")
    print(f"{'='*60}")
    print(interpretation)
    print(f"{'='*60}")
    
    # 详细数据
    print(f"\n上涨能量序列 (最近5个): {result['up_sequence']}")
    print(f"下跌能量序列 (最近5个): {result['down_sequence']}")
    print(f"上涨趋变系数 ʋ: {result['up_v']:.4f} ({result['up_trend']})")
    print(f"下跌趋变系数 ʋ: {result['down_v']:.4f} ({result['down_trend']})")
    
    # 保存JSON
    if '--json' in sys.argv:
        with open('tide_state_signal.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n信号已保存: tide_state_signal.json")
    
    # 回看历史平衡信号
    if '--history' in sys.argv:
        print(f"\n历史状态翻转信号 (优势度接近0):")
        n = len(df)
        for t in range(100, n, 20):  # 每20天采样
            sub_small = [s for s in small if s['end_idx'] <= t and s['end_idx'] >= t - 90]
            if len(sub_small) >= 4:
                r = analyze_tide_state(pd.DataFrame({
                    'Close': df['Close'].values[:t+1],
                    'Date': df['Date'].values[:t+1],
                }).assign(High=lambda x: x['Close'], Low=lambda x: x['Close']), 
                    sub_small)
                
                if r.get('reversal_signal') and r['signal_strength'] > 0.8:
                    date_str = str(df['Date'].values[t])
                    print(f"  {date_str[:10]} price={df['Close'].values[t]:.0f} adv={r['state_advantage']:+.3f} {r['signal_detail'][:40]}")
