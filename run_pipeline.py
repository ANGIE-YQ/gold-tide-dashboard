"""
run_pipeline.py —— 黄金潮汐模型统一流水线
============================================================
一键运行完整分析: 潮汐分段 -> 特征构建 -> ML验证 -> 风险叠加 -> 交易信号

用法:
    python run_pipeline.py              # 完整运行
    python run_pipeline.py --quick      # 仅信号简报
    python run_pipeline.py --rebuild    # 重建特征层+完整运行

输出:
    - 控制台: 信号简报 + 交易建议
    - deliverables/investment-masters/AU0-gold-tidal-analysis-YYYY-MM-DD.md
============================================================
"""
import sys, os, argparse
import numpy as np
import pandas as pd

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'gold_AU0_daily.csv')
FEAT_PATH = os.path.join(BASE_DIR, 'gold_features.csv')
OUT_DIR = os.path.join(BASE_DIR, 'deliverables', 'investment-masters')


def phase1_engine():
    """Phase 1: 潮汐分段引擎"""
    from gold_tide_engine import load_data, detect_tides, compute_atr
    print('[Phase 1] 潮汐分段引擎...')
    df = load_data(DATA_PATH)
    atr = compute_atr(df, 20)
    big, small, big_pv, small_pv = detect_tides(df, 3.0, 1.0, atr)
    print(f'  大潮汐: {len(big)} 段  小潮汐: {len(small)} 段')
    print(f'  数据: {df["Date"].min().date()} ~ {df["Date"].max().date()}  ({len(df)} 根)')
    return df, atr, big, small, big_pv, small_pv


def phase2_features(df, atr, big, small, small_pv, force_rebuild=False):
    """Phase 2: 特征层构建"""
    from gold_tide_score import score_all_tides
    from gold_tide_features import build_features
    if os.path.exists(FEAT_PATH) and not force_rebuild:
        print('[Phase 2] 特征层已存在,跳过重建 (--rebuild 强制重建)')
        feat = pd.read_csv(FEAT_PATH, parse_dates=['Date'])
        return feat
    print('[Phase 2] 构建无前视特征层...')
    score_all_tides(df, big, small, atr)
    feat = build_features(df, atr, big, small, small_pv, save_csv=True)
    print(f'  特征矩阵: {feat.shape[0]} 行 x {feat.shape[1]} 列')
    return feat


def phase3_ml_validation(feat):
    """Phase 3: ML Walk-Forward 验证"""
    from gold_tide_ml import load_features, walk_forward
    print('[Phase 3] XGBoost Walk-Forward 验证...')
    feat_df = load_features(FEAT_PATH)

    results = {}
    for fwd in (5, 10, 20):
        r = walk_forward(feat_df, fwd)
        results[fwd] = r
        print(f'  fwd={fwd:2d}: AUC={r["mean_auc"]:.3f}  '
              f'Acc={r["mean_acc"]:.1%}  CAcc={r["mean_cacc"]:.1%}  '
              f'folds={r["n_folds"]}')

    return results


def phase4_t3_risk():
    """Phase 4: T3 风险叠加策略"""
    from gold_tide_t3 import main as t3_main
    print('[Phase 4] T3 风险叠加策略...')
    # T3 有独立入口,直接调用其 main
    # 这里只做轻量调用
    from gold_tide_engine import load_data, compute_atr
    from gold_tide_t3 import gen_p, strategy_equity, _warm
    df = load_data(DATA_PATH)
    atr_full = compute_atr(df, 20)
    feat = pd.read_csv(FEAT_PATH, parse_dates=['Date'])
    warm = _warm(df, atr_full)
    close = df['Close'].values.astype(float)[warm:]
    atr = atr_full[warm:]
    sma = pd.Series(close).rolling(250, min_periods=50).mean().values
    prob = gen_p(feat, 10)

    for cost in (0.0005, 0.002):
        r_t3 = strategy_equity(close, atr, sma, feat, prob, 't3', cost)
        print(f'  成本{cost*100:.2f}%: 年化={r_t3["ann"]*100:.1f}%  '
              f'夏普={r_t3["sharpe"]:.2f}  回撤={r_t3["mdd"]*100:.1f}%')

    return prob


def phase5_signal(feat, prob):
    """Phase 5: 当前交易信号（全量训练模型预测当前点）"""
    from gold_tide_engine import load_data, compute_atr, detect_tides
    from gold_tide_score import score_all_tides
    from xgboost import XGBClassifier
    from gold_tide_ml import FEATURES
    print('[Phase 5] 生成当前交易信号...')

    df = load_data(DATA_PATH)
    close = df['Close'].values.astype(float)
    dates = df['Date'].values.astype('datetime64[D]')
    n = len(close)

    # 全量训练模型,预测当前bar
    feats = [c for c in feat.columns if c not in ('Date', 'fwd5', 'fwd10', 'fwd20', 'Close')]
    y = (feat['fwd10'] > 0).astype(int).values
    valid = ~feat['fwd10'].isna().values
    X = feat[feats].values

    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42
    )
    model.fit(X[valid], y[valid])

    # 预测当前
    cur_X = X[-1:].reshape(1, -1)
    p_up = float(model.predict_proba(cur_X)[0, 1])

    # 还检查walk-forward的概率（如果有）
    wf_prob = prob.get(len(feat) - 1, None)

    # 特征重要性
    imp = sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1])[:5]

    # 当前潮汐状态
    atr_arr = compute_atr(df, 20)
    big, small, _, _ = detect_tides(df, 3.0, 1.0, atr_arr)
    score_all_tides(df, big, small, atr_arr)

    cur_tide = big[-1]
    ip_small = [s for s in small if s['start_idx'] >= cur_tide['end_idx']]

    direction = 'BUY' if p_up > 0.55 else ('SELL' if p_up < 0.45 else 'HOLD')
    conf = abs(p_up - 0.5)

    signal = {
        'date': str(dates[-1]),
        'price': close[-1],
        'p_up': p_up,
        'wf_p_up': wf_prob,
        'direction': direction,
        'confidence': conf,
        'last_tide': cur_tide['tide_id'],
        'last_tide_dir': cur_tide['direction'],
        'last_tide_score': cur_tide.get('momentum_score', 50),
        'ip_days': n - cur_tide['end_idx'] if cur_tide['end_idx'] < n else 0,
        'ip_small_count': len(ip_small),
        'atr': atr_arr[-1],
        'ma20': np.mean(close[-20:]),
        'high20': close[-20:].max(),
        'low20': close[-20:].min(),
        'top_features': imp,
    }

    # 止损/止盈
    atr = signal['atr']
    price = signal['price']
    if signal['direction'] == 'BUY':
        signal['stop_loss'] = price - 2.5 * atr
        signal['target'] = price + 3.0 * atr
    elif signal['direction'] == 'SELL':
        signal['stop_loss'] = price + 2.5 * atr
        signal['target'] = price - 3.0 * atr
    else:
        signal['stop_loss'] = price - 2 * atr
        signal['target'] = price + 2 * atr

    return signal


def generate_report(ml_results, t3_prob, signal, output_path):
    """生成Markdown分析报告"""
    today = signal['date']

    report = f'''# 黄金潮汐模型 · 分析报告

**生成日期**: {today}
**分析标的**: 沪金主力 AU0 日线
**数据来源**: NeoData 金融数据服务
**模型架构**: 潮汐分段引擎 + 31维无前视特征层 + XGBoost Walk-Forward + T3风险叠加

---

## 一、当前交易信号

| 项目 | 内容 |
|------|------|
| **方向** | **{signal['direction']}** |
| **P(涨)** | {signal['p_up']:.3f} |
| **置信度** | {signal['confidence']:.3f} |
| **当前价** | {signal['price']:.2f} |
| **止损** | {signal['stop_loss']:.0f} |
| **目标** | {signal['target']:.0f} |
| **ATR** | {signal['atr']:.2f} |

### 潮汐状态

| 项目 | 内容 |
|------|------|
| 最后潮汐 | {signal['last_tide']} ({'↑' if signal['last_tide_dir']=='up' else '↓'}) |
| 动能分 | {signal['last_tide_score']:.1f}/100 |
| 进行中 | {signal['ip_days']} 日, {signal['ip_small_count']} 段小潮汐 |
| 20日均 | {signal['ma20']:.0f} |
| 20日区间 | {signal['low20']:.0f} ~ {signal['high20']:.0f} |

---

## 二、ML Walk-Forward 验证

| Horizon | AUC | 方向准确率 | 置信子集准确率 | 折叠数 |
|---------|-----|-----------|---------------|--------|
'''
    for fwd in (5, 10, 20):
        r = ml_results[fwd]
        report += f'| fwd={fwd} | {r["mean_auc"]:.3f} | {r["mean_acc"]:.1%} | {r["mean_cacc"]:.1%} | {r["n_folds"]} |\n'

    report += f'''
---

## 三、风险提示

1. **短期信号**: 模型预测周期为未来10个交易日,不适用于长期持仓
2. **单市场风险**: 仅基于沪金AU0日线训练,未纳入宏观因子(利率/美元)
3. **尾部风险**: Walk-Forward OOS AUC {ml_results[10]["mean_auc"]:.2f} 表明模型有统计优势,但单次预测仍可能错误
4. **成本假设**: 回测基于0.05%/边成本,实盘滑点可能更高

---

⚠️ 以上内容由 AI 基于历史价格形态统计模型生成,仅供参考,不构成任何投资建议。
'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


def print_banner(signal, ml_results):
    """打印信号简报"""
    print()
    print('=' * 64)
    print('  黄金潮汐模型 · 交易信号')
    print('=' * 64)
    p = signal['p_up']
    conf = signal['confidence']
    bar_len = 30
    bar = '█' * int(p * bar_len) + '░' * (bar_len - int(p * bar_len))

    print(f'''
  日期: {signal['date']}    价格: {signal['price']:.2f}

  P(涨) = {p:.3f}  [{bar}]
  WF参考: {signal.get("wf_p_up", "N/A")}

  方向:  {'▲ BUY' if signal['direction']=='BUY' else ('▼ SELL' if signal['direction']=='SELL' else '─ HOLD')}
  置信:  {'强' if conf > 0.15 else ('中' if conf > 0.08 else '弱')} ({conf:.3f})

  止损:  {signal['stop_loss']:.0f}    目标:  {signal['target']:.0f}
  仓位:  {'15-20%' if conf > 0.15 else ('8-12%' if conf > 0.08 else '观望/轻仓')}

  ML验证:  fwd10 AUC={ml_results[10]['mean_auc']:.2f}  Acc={ml_results[10]['mean_acc']:.1%}
  特征top3: {', '.join(f'{k}({v:.3f})' for k,v in signal.get('top_features', [])[:3])}
''')
    print('=' * 64)
    print('⚠️ 以上内容由统计模型生成,仅供参考,不构成投资建议.')


def main():
    parser = argparse.ArgumentParser(description='黄金潮汐模型统一流水线')
    parser.add_argument('--quick', action='store_true', help='仅输出信号简报')
    parser.add_argument('--rebuild', action='store_true', help='强制重建特征层')
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    # Phase 1
    df, atr, big, small, big_pv, small_pv = phase1_engine()

    if args.quick:
        # 快速模式:直接使用已有特征层
        if not os.path.exists(FEAT_PATH):
            print('特征层不存在,先构建...')
            phase2_features(df, atr, big, small, small_pv, force_rebuild=True)
        feat = pd.read_csv(FEAT_PATH, parse_dates=['Date'])
        from gold_tide_t3 import gen_p
        prob = gen_p(feat, 10)
        from gold_tide_ml import load_features, walk_forward
        feat_df = load_features(FEAT_PATH)
        ml_results = {10: walk_forward(feat_df, 10)}
    else:
        # Phase 2
        feat = phase2_features(df, atr, big, small, small_pv, force_rebuild=args.rebuild)
        # Phase 3
        ml_results = phase3_ml_validation(feat)
        # Phase 4
        prob = phase4_t3_risk()
        # Phase 5
        signal = phase5_signal(feat, prob)
        # 输出
        print_banner(signal, ml_results)
        # 落盘
        today = signal['date']
        outpath = os.path.join(OUT_DIR, f'AU0-gold-tidal-analysis-{today}.md')
        generate_report(ml_results, prob, signal, outpath)
        print(f'\n完整报告: {outpath}')


if __name__ == '__main__':
    main()
