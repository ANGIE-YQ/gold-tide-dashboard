#!/usr/bin/env python3
"""
多模型训练 & 对比管线
=====================
v1: 原始31维 (baseline)
v2: 简化18维 (潮汐核心+拐点, 最优)
v3: 全量55维 (31+24 ±En·ʋ特征, 历史最佳)
"""

import numpy as np, pandas as pd, pickle, json, os, sys
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)


# ============================================================
# 1. 确保特征就绪 (含±En·ʋ)
# ============================================================
def ensure_features():
    """如果特征文件不含 ±En·ʋ 列则重建"""
    feat = pd.read_csv('gold_features_enhanced.csv', parse_dates=['Date'])
    has_en_v = any(c.startswith('en_v_') for c in feat.columns)
    if not has_en_v:
        print('[特征] 重建 ±En·ʋ 特征...')
        from gold_tide_en_v import build_en_v_features
        feat = build_en_v_features()
    else:
        n_en_v = sum(1 for c in feat.columns if c.startswith('en_v_'))
        print(f'[特征] 已有 {n_en_v} 维 ±En·ʋ 特征, 共 {feat.shape[1]} 列')
    return feat


# ============================================================
# 2. 特征集定义
# ============================================================
def define_feature_sets(feat):
    """定义三个模型的特征集"""
    skip = {'Date', 'Close', 'fwd5', 'fwd10', 'fwd20',
            'vol_label', 'opt_atr_label'}
    dead = {'ip_small_cnt', 'ip_small_up_ratio',
            'ip_last_small_dir', 'ip_trend_slope'}

    all_cols = [c for c in feat.columns if c not in skip and c not in dead]

    # v1: 原始31维 (与旧版一致)
    v1_features = [c for c in all_cols
                   if not c.startswith('en_v_')
                   and c not in {'vol_label', 'opt_atr_label'}]

    # v2: 简化18维 (仅潮汐核心+拐点, benchmark验证最优)
    v2_features = [c for c in v1_features
                   if c.startswith(('cur_', 'prev_', 'lp_'))]

    # v3: 全量55维
    v3_features = all_cols

    return {
        'v1_原始31维': v1_features,
        'v2_简化18维': v2_features,
        'v3_全量55维': v3_features,
    }


# ============================================================
# 3. 训练+校准+保存
# ============================================================
def train_model(feat, features, name):
    """训练XGBoost+Platt校准, 返回模型对象"""
    print(f'[训练] {name} ({len(features)}维)...', end=' ')

    label_col = 'fwd10'
    y = (feat[label_col] > 0).astype(int).values
    ok = ~feat[label_col].isna().values
    X = feat[features].values

    base = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42)
    base.fit(X[ok], y[ok])
    cal = CalibratedClassifierCV(base, method='sigmoid',
                                  cv=TimeSeriesSplit(n_splits=5))
    cal.fit(X[ok], y[ok])

    # 特征重要性
    imp = sorted(zip(features, base.feature_importances_),
                 key=lambda x: -x[1])

    # 预测评估
    probs = cal.predict_proba(X[ok])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[ok], probs)

    model_pkg = {
        'model': cal,
        'features': features,
        'dead': list(dead_cols(feat)),
        'feature_importance': imp[:15],
        'auc': round(float(auc), 4),
        'n_features': len(features),
        'n_samples': int(sum(ok)),
        'train_date': str(feat['Date'].max().date()),
        'version': name,
    }

    # 保存
    path = f'gold_tide_model_{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_pkg, f)

    print(f'AUC={auc:.3f} 保存→{path}')
    return model_pkg


def dead_cols(feat):
    return {'ip_small_cnt', 'ip_small_up_ratio',
            'ip_last_small_dir', 'ip_trend_slope'}


# ============================================================
# 4. 回测对比
# ============================================================
def run_benchmarks():
    """三个模型全阈值回测对比"""
    from model_benchmark import BacktestEngine

    feat = pd.read_csv('gold_features_enhanced.csv', parse_dates=['Date'])
    eng = BacktestEngine()

    thresholds = [
        (0.65, 0.35, '保守'),
        (0.60, 0.40, '标准'),
        (0.55, 0.45, '积极'),
    ]

    models = [
        ('v1_原始31维', 'gold_tide_model_v1_原始31维.pkl'),
        ('v2_简化18维', 'gold_tide_model_v2_简化18维.pkl'),
        ('v3_全量55维', 'gold_tide_model_v3_全量55维.pkl'),
    ]

    results = []
    for name, path in models:
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            pkg = pickle.load(f)
        fc = [c for c in pkg['features'] if c in feat.columns]

        for bt, st, tl in thresholds:
            r = eng.run(f'{name}_{tl}', fc, (bt, st))
            if r:
                r['model'] = name
                r['threshold'] = tl
                results.append(r)

    return results


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print('=' * 60)
    print('  多模型训练 & 对比管线')
    print('=' * 60)

    feat = ensure_features()
    sets = define_feature_sets(feat)

    # 训练三个模型
    models = {}
    for name, features in sets.items():
        n_avail = len([c for c in features if c in feat.columns])
        models[name] = train_model(feat, features, name)

    # 回测对比
    print(f'\n{"=" * 60}')
    print('  回测对比')
    print('=' * 60)
    results = run_benchmarks()

    print(f'\n{"模型":<16} {"阈值":<6} {"夏普":>6} {"收益":>9} {"胜率":>5} {"交易":>5} {"回撤":>6} {"AUC":>6}')
    print('-' * 70)
    for r in results:
        print(f'{r["model"]:<16} {r["threshold"]:<6} '
              f'{r["sharpe"]:>6.2f} {r["total_return_pct"]:>+8.1f}% '
              f'{r["win_rate_pct"]:>4.0f}% {r["n_trades"]:>4}笔 '
              f'{r["max_dd_pct"]:>5.1f}% '
              f'{models.get(r["model"], {}).get("auc", "?"):>6}')

    # 保存对比结果
    with open('model_comparison_v2.json', 'w', encoding='utf-8') as f:
        json.dump([{k: v for k, v in r.items() if k != 'model'}
                   for r in results], f, ensure_ascii=False, indent=2)

    print(f'\n结果: model_comparison_v2.json')
    print(f'模型文件: gold_tide_model_v*.pkl (3个)')

    # AUCD对比
    print(f'\n  模型质量对比:')
    for name, pkg in models.items():
        print(f'    {name}: AUC={pkg["auc"]:.4f}  '
              f'{pkg["n_features"]}维 {pkg["n_samples"]}样本')


if __name__ == '__main__':
    main()
