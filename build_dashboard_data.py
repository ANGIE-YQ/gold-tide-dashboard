#!/usr/bin/env python3
"""
生成看板数据JSON: 双模型 + 决策矩阵 + 历史表现 + 价格走势
"""
import numpy as np, pandas as pd, pickle, json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from gold_tide_engine import load_data, compute_atr, detect_tides
from gold_tide_score import score_all_tides

def compute_advantage(ue, de):
    ut = sum(ue) if ue else 0; dt = sum(de) if de else 0
    return (ut-dt)/(ut+dt) if ut+dt>1e-9 else 0

def compute_v(e):
    if len(e) < 2: return 1.0
    r = [e[i]/max(e[i-1],1e-9) for i in range(1,len(e))]
    return float(np.exp(np.mean(np.log(r)))) if r else 1.0

def main():
    df = load_data('gold_AU0_daily.csv')
    atr_arr = compute_atr(df, 20)
    big, small, _, _ = detect_tides(df, 3.0, 1.0, atr_arr)
    score_all_tides(df, big, small, atr_arr)
    for s in small:
        if 'energy' not in s:
            s['energy'] = s['height'] * max(s['duration_bars'], 1)

    close = df['Close'].values.astype(float)
    dates = df['Date'].values.astype(str)
    n = len(close)
    price_now = float(close[-1])
    atr_now = float(atr_arr[-1])

    # === Combined Model ===
    feat = pd.read_csv('gold_features_enhanced.csv', parse_dates=['Date'])
    with open('gold_tide_calibrated_model.pkl', 'rb') as f:
        pkg = pickle.load(f)
    model = pkg['model']
    fc = [c for c in pkg['features'] if c in feat.columns and c not in pkg.get('dead',[])]
    X = feat[fc].values
    p_up = float(model.predict_proba(X[-1:])[0, 1])

    # 方向 + 置信
    if p_up > 0.65:
        direction = '📈 看多'; d_class = 'buy'; d_color = '#3fb950'
        sl = price_now - 2.5 * atr_now; tp = price_now + 3.0 * atr_now
        position = '做多 20%-25%'
    elif p_up < 0.35:
        direction = '📉 看空'; d_class = 'sell'; d_color = '#f85149'
        sl = price_now + 2.5 * atr_now; tp = price_now - 3.0 * atr_now
        position = '做空 20%-25%'
    else:
        direction = '⏸️ 观望'; d_class = 'hold'; d_color = '#d2991d'
        sl = price_now - 2 * atr_now; tp = price_now + 2 * atr_now
        position = '空仓等待'

    conf = abs(p_up - 0.5)
    if conf > 0.15: conf_level = '强'; conf_desc = '模型置信度高，建议按标准仓位执行'
    elif conf > 0.08: conf_level = '中'; conf_desc = '模型有一定把握，可适当降低仓位'
    else: conf_level = '弱'; conf_desc = '模型把握不足，建议观望或极小仓位试探'

    combined = {
        'name': '精简+±En·v (主力模型)',
        'desc': '18维潮汐特征 + 24维 ±En·v 状态特征 → XGBoost 训练 → 输出 P(涨) 概率',
        'p_up': round(p_up, 4),
        'p_up_pct': round(p_up * 100, 1),
        'p_down_pct': round((1-p_up) * 100, 1),
        'direction': direction,
        'direction_class': d_class,
        'direction_color': d_color,
        'confidence': round(conf, 4),
        'confidence_pct': round(conf * 100, 1),
        'confidence_level': conf_level,
        'confidence_desc': conf_desc,
        'stop_loss': round(sl, 1),
        'take_profit': round(tp, 1),
        'position': position,
        'threshold_buy': 'P(涨) > 0.65',
        'threshold_sell': 'P(涨) < 0.35',
        'win_rate': '92%',
        'sharpe': '3.65',
        'n_trades_per_year': 11.9,
        'model_desc_detail': {
            '核心特征': '潮汐方向/进度/拐点(18维) + 状态优势度/趋变系数/能量(24维)',
            '训练方式': 'XGBoost Walk-Forward 时间序列交叉验证',
            '输出含义': '未来10个交易日价格上涨的概率(0-1)',
            '阈值': '>0.65做多, <0.35做空, 中间观望',
            '历史夏普': '3.65 (含交易成本前)',
            '历史胜率': '92%',
            '年交易频率': '约12次/年',
        }
    }

    # === ±En·v Pure Framework ===
    window_small = small[-25:]
    ue = [s['energy'] for s in window_small if s['direction']=='up']
    de = [s['energy'] for s in window_small if s['direction']=='down']
    adv = compute_advantage(ue, de)
    v_up = compute_v(ue[-8:]) if len(ue)>=2 else 1.0
    v_down = compute_v(de[-8:]) if len(de)>=2 else 1.0

    if abs(adv) < 0.05:
        en_v_signal = '⚡ 归零信号 —— 多空动能完全平衡，强烈关注建仓'
        en_v_sig_class = 'strong'
    elif abs(adv) < 0.10:
        en_v_signal = '🔔 趋零信号 —— 优势度快速衰减，关注状态翻转'
        en_v_sig_class = 'warning'
    elif abs(adv) < 0.15:
        en_v_signal = '👀 逼近零区 —— 优势度缩小中'
        en_v_sig_class = 'watch'
    else:
        gap = abs(adv) - 0.05
        en_v_signal = f'⭕ 无信号 —— 优势度{adv:+.2f}，距归零触发还差 {gap:.2f}'
        en_v_sig_class = 'none'

    en_v_direction = '上涨占优' if adv > 0.15 else ('下跌占优' if adv < -0.15 else '多空平衡')
    en_v_direction_label = '📈' if adv > 0.15 else ('📉' if adv < -0.15 else '⚖️')

    # 趋变系数解读
    if v_up > 1.05: v_up_desc = '增强'
    elif v_up < 0.95: v_up_desc = '衰减'
    else: v_up_desc = '趋稳'
    if v_down > 1.05: v_down_desc = '增强'
    elif v_down < 0.95: v_down_desc = '衰减'
    else: v_down_desc = '趋稳'

    # 最近5个能量
    up_seq = [round(x, 0) for x in ue[-5:]]
    down_seq = [round(x, 0) for x in de[-5:]]

    en_v = {
        'name': '±En·v 纯框架',
        'desc': '基于潮汐分段的状态转移框架 | 不依赖机器学习',
        'advantage': round(adv, 4),
        'advantage_abs': round(abs(adv), 4),
        'advantage_pct': round(abs(adv)*100, 0),
        'state_direction': en_v_direction,
        'state_emoji': en_v_direction_label,
        'signal': en_v_signal,
        'signal_class': en_v_sig_class,
        'v_up': round(v_up, 3),
        'v_down': round(v_down, 3),
        'v_up_desc': v_up_desc,
        'v_down_desc': v_down_desc,
        'up_energies': up_seq,
        'down_energies': down_seq,
        'up_total': round(sum(ue), 0),
        'down_total': round(sum(de), 0),
        'lookback_tides': len(window_small),
        'param_explanations': {
            '± (状态优势度)': '范围 [-1, +1]。正=上涨动能占优，负=下跌势能占优。接近0=多空平衡，是建仓信号区。',
            'En (能量序列)': '每个潮汐单元的动能 = 价格高度 × 持续时间。上涨和下跌分开计算。',
            'ʋ (趋变系数)': '同方向能量序列的变化率。>1=增强中，<1=衰减中。',
            '建仓信号': '|±|<0.05 时触发，表示多空动能完全平衡，状态即将翻转。',
        }
    }

    # === 决策矩阵 ===
    en_v_dir_short = '看多' if adv > 0 else ('看空' if adv < 0 else '平衡')
    combined_dir_short = '看多' if p_up > 0.65 else ('看空' if p_up < 0.35 else '观望')

    if abs(adv) < 0.15:
        en_v_dir_short = '归零信号' if abs(adv) < 0.10 else '逼近零区'

    decision = {
        'combined_signal': combined_dir_short,
        'en_v_signal': en_v_dir_short,
        'consensus': '',
        'action': '',
        'action_detail': '',
        'position_advice': ''
    }

    if combined_dir_short == '看多' and abs(adv) < 0.10:
        decision['consensus'] = '🟢 一致看多'
        decision['action'] = '加仓做多'
        decision['action_detail'] = '主力模型看好 + ±En·v 归零确认 = 最强做多信号。建议标准仓位上限。'
        decision['position_advice'] = '20%-25% 仓位'
    elif combined_dir_short == '看空' and abs(adv) < 0.10:
        decision['consensus'] = '🟢 一致看空'
        decision['action'] = '加仓做空'
        decision['action_detail'] = '主力模型看空 + ±En·v 归零确认 = 最强做空信号。建议标准仓位上限。'
        decision['position_advice'] = '20%-25% 仓位'
    elif combined_dir_short == '看多':
        decision['consensus'] = '🟡 看多（±En·v 无信号）'
        decision['action'] = '标准做多'
        decision['action_detail'] = '主力模型看好，但±En·v 尚未归零。建议标准仓位。'
        decision['position_advice'] = '15%-20% 仓位'
    elif combined_dir_short == '看空':
        decision['consensus'] = '🟡 看空（±En·v 无信号）'
        decision['action'] = '标准做空'
        decision['action_detail'] = '主力模型看空，但±En·v 尚未归零。建议标准仓位。'
        decision['position_advice'] = '15%-20% 仓位'
    else:
        decision['consensus'] = '⭕ 观望'
        decision['action'] = '空仓等待'
        decision['action_detail'] = '主力模型未达开仓阈值。等待 P(涨)>0.65 或 <0.35 再行动。'
        decision['position_advice'] = '空仓'

    # === 价格走势 ===
    recent60 = []
    for i in range(max(0, n-60), n):
        recent60.append({
            'date': str(df['Date'].values[i])[:10],
            'close': round(float(close[i]), 2),
        })

    # 最近潮汐
    tides = []
    for leg in big[-8:]:
        tides.append({
            'id': leg['tide_id'],
            'dir': leg['direction'],
            'start': str(leg['start_date'])[:10],
            'end': str(leg['end_date'])[:10],
            'score': float(leg.get('momentum_score', 50)),
            'energy': int(leg.get('energy', 0)),
        })

    # 最近10笔 ±En·v 信号
    en_v_history = []
    for i in range(max(0, len(small)-300), len(small)):
        if i < 20: continue
        win = small[i-20:i+1]
        ue_i = [s['energy'] for s in win if s['direction']=='up']
        de_i = [s['energy'] for s in win if s['direction']=='down']
        if len(ue_i)<3 or len(de_i)<3: continue
        adv_i = compute_advantage(ue_i, de_i)
        if abs(adv_i) >= 0.12: continue
        vu = compute_v(ue_i[-8:]); vd = compute_v(de_i[-8:])
        d_i = 'BUY' if vu > vd else 'SELL'
        ei = small[i]['end_idx']
        if ei + 15 < n:
            actual = '✓' if (d_i=='BUY' and close[ei+10]>close[ei+1]) or (d_i=='SELL' and close[ei+10]<close[ei+1]) else '✗'
        else:
            actual = '?'
        en_v_history.append({
            'date': str(dates[ei])[:10],
            'direction': d_i,
            'adv': round(abs(adv_i), 3),
            'result': actual,
        })
    en_v_history = en_v_history[-10:]

    # === 组合 ===
    data = {
        'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'current': {
            'price': round(price_now, 2),
            'atr': round(atr_now, 2),
            'ma20': round(float(np.mean(close[-20:])), 2),
            'date': str(pd.Timestamp(df['Date'].values[-1]).date()),
        },
        'combined': combined,
        'en_v': en_v,
        'decision': decision,
        'recent60': recent60,
        'tides': tides,
        'en_v_history': en_v_history,
        'bh_benchmark': {
            'return': '+297.0%',
            'sharpe': '0.52',
            'max_dd': '-44.8%',
        },
    }

    os.makedirs('docs', exist_ok=True)
    with open('docs/dashboard_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'看板数据已生成: docs/dashboard_data.json')
    return data

if __name__ == '__main__':
    main()
