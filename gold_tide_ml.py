"""
gold_tide_ml.py —— 潮汐特征层 → 树模型（T1 核心）
扩展窗口 walk-forward：训练用全部历史，测试用滚动未来段；严格无前视。
主指标：OOS AUC / 方向准确率 / 置信子集准确率；并输出特征重要性。
对比基线：多数类基线、朴素 V0 规则(~50%)。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURES = None


def load_features(path='D:/Work/module/gold_features.csv'):
    global FEATURES
    df = pd.read_csv(path, parse_dates=['Date'])
    FEATURES = [c for c in df.columns if c not in ('Date', 'fwd5', 'fwd10', 'fwd20')]
    return df


def walk_forward(df, fwd, test_len=400, train_init=1500, conf=0.15):
    y = (df['fwd%d' % fwd] > 0).astype(int).values
    valid = np.where(~df['fwd%d' % fwd].isna().values)[0]
    n_val = len(valid)
    X = df[FEATURES].values
    pos = train_init
    aucs, accs, caccs, covs, nte = [], [], [], [], []
    all_true, all_prob = [], []
    while pos + test_len <= n_val:
        tr = valid[:pos]; te = valid[pos:pos + test_len]
        if len(tr) < 300 or len(te) < 50:
            break
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            eval_metric='logloss', random_state=42)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        yt = y[te]
        auc = roc_auc_score(yt, p)
        acc = float(np.mean((p > 0.5) == yt))
        mask = np.abs(p - 0.5) > conf
        cacc = float(np.mean((p[mask] > 0.5) == yt[mask])) if mask.sum() > 0 else float('nan')
        aucs.append(auc); accs.append(acc); caccs.append(cacc); covs.append(mask.mean()); nte.append(len(te))
        all_true.extend(yt.tolist()); all_prob.extend(p.tolist())
        pos += test_len
    overall_auc = roc_auc_score(all_true, all_prob)
    return dict(overall_auc=overall_auc, mean_auc=np.mean(aucs), mean_acc=np.mean(accs),
                mean_cacc=np.nanmean(caccs), mean_cov=np.mean(covs), n_folds=len(aucs),
                accs=accs, aucs=aucs, caccs=caccs, nte=nte)


def main():
    df = load_features()
    print('=' * 72)
    print('潮汐特征层 → XGBoost  walk-forward（扩展窗口，严格无前视）')
    print('=' * 72)
    # 基线：多数类
    base = {}
    for fwd in (5, 10, 20):
        y = (df['fwd%d' % fwd] > 0).astype(int)
        base[fwd] = float(y.mean())
    print('标签正向占比(多数类基线准确率): fwd5=%.3f fwd10=%.3f fwd20=%.3f' % (
        base[5], base[10], base[20]))
    print('（朴素 V0 规则方向准确率≈50%%，作为另一基线对照）\n')

    for fwd in (5, 10, 20):
        r = walk_forward(df, fwd)
        print('[fwd=%2d] 折叠数=%d  平均OOS AUC=%.3f  整体AUC=%.3f' % (
            fwd, r['n_folds'], r['mean_auc'], r['overall_auc']))
        print('   方向准确率=%5.1f%%  置信子集(覆盖%.0f%%)准确率=%5.1f%%' % (
            100 * r['mean_acc'], 100 * r['mean_cov'], 100 * r['mean_cacc']))
        print('   各折叠AUC =', [round(x, 3) for x in r['aucs']])
        print('   各折叠准确率 =', [round(100 * x, 1) for x in r['accs']])

    # 特征重要性（全样本训练一次，供参考：哪些潮汐特征真被模型用上）
    print('\n特征重要性（全样本 XGBoost，仅供参考结构价值）:')
    y = (df['fwd10'] > 0).astype(int).values
    clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        eval_metric='logloss', random_state=42)
    clf.fit(df[FEATURES].values, y)
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
    for k, v in imp[:12]:
        print('   %-16s %.3f' % (k, v))


if __name__ == '__main__':
    main()
