"""
gold_tide_t3.py —— T3 风险叠加：把 T1 已验证信号做成"带风险预算的可部署策略"
组件：
  · 信号: XGBoost P(涨 fwd=10) → 方向=sign(p-0.5)
  · 置信加权 w_conf: |p-0.5| 越大仓位越满
  · 动能约束调制 w_mom: 当前潮汐通道越窄(强约束)仓位越满
  · vol-targeting w_vol: 目标日波动1%，按 ATR/价 缩放
  · 牛熊 regime: 熊市仓位×0.7
  · ATR 止损: 浮亏超 2×ATR 平仓
对比：裸信号(仅方向) / 买持有；成本 0.05% 与 0.20% 两档。
信号 p 由扩展窗口 walk-forward 产出（严格无前视）。
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from gold_tide_engine import load_data, compute_atr, CONFIG

TRAIN_INIT, TEST_LEN = 1500, 400
TARGET_VOL = 0.01


def _warm(df, atr):
    from gold_tide_engine import detect_tides
    big, _, _, _ = detect_tides(df, 3, 1, atr)
    return int(big[0]['start_idx']) + 20


def gen_p(df_feat, fwd=10):
    feats = [c for c in df_feat.columns if c not in ('Date', 'fwd5', 'fwd10', 'fwd20', 'Close')]
    y = (df_feat['fwd%d' % fwd] > 0).astype(int).values
    valid = np.where(~df_feat['fwd%d' % fwd].isna().values)[0]
    X = df_feat[feats].values
    pos = TRAIN_INIT
    prob = {}
    while pos + TEST_LEN <= len(valid):
        tr = valid[:pos]; te = valid[pos:pos + TEST_LEN]
        if len(tr) < 300 or len(te) < 50:
            break
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                            eval_metric='logloss', random_state=42)
        clf.fit(X[tr], y[tr])
        for idx, pp in zip(te, clf.predict_proba(X[te])[:, 1]):
            prob[int(idx)] = float(pp)
        pos += TEST_LEN
    return prob


def strategy_equity(close, atr, sma, feat, prob, mode, cost):
    """mode: 'raw' 仅方向无止损 | 'naked' 方向+ATR止损 | 't3' 风险叠加+止损。
    返回 (total, ann, sharpe, mdd, n)。close/atr/sma 已按特征行对齐。"""
    n = len(close)
    idxs = sorted(prob.keys()); idx_set = set(idxs)
    ret = np.zeros(n)
    pos = 0.0
    entry_px = 0.0
    entry_atr = 0.0
    use_stop = (mode != 'raw')
    for t in idxs:
        if t + 1 >= n:
            continue
        p = prob[t]
        sig = 1.0 if p > 0.5 else -1.0
        desired = sig
        if mode == 't3':
            # 风险预算(改进版)：高置信才出手(过滤)，温和动能调制，vol-targeting，牛熊缩放
            confident = abs(p - 0.5) > 0.08
            w_mom = 0.7 + 0.3 * float(feat.iloc[t]['cur_band_norm'])      # 0.7..1.0 温和
            w_vol = float(np.clip(TARGET_VOL / (atr[t] / (close[t] + 1e-9)), 0.6, 1.4))  # vol-target
            regime = 1.0 if close[t] > sma[t] else 0.85
            desired = float(np.clip(sig * (1.0 if confident else 0.0) * w_mom * w_vol * regime, -1, 1))
        # ATR 止损：持仓中且浮亏超 2×ATR → 平仓
        eff = desired
        if use_stop and pos != 0:
            adverse = (pos > 0 and close[t] < entry_px - 2 * entry_atr) or \
                      (pos < 0 and close[t] > entry_px + 2 * entry_atr)
            if adverse:
                eff = 0.0
        if eff != 0 and pos == 0:
            entry_px = close[t]; entry_atr = atr[t]
        elif eff == 0:
            pos = 0.0
        day = close[t + 1] / close[t] - 1.0
        tc = cost * abs(eff - pos)
        ret[t] = eff * day - tc
        pos = eff
    eq = np.cumprod(1 + ret)
    mask = np.array([t in idx_set for t in range(n)])
    sub = eq[mask]; rsub = ret[mask]
    yrs = mask.sum() / 252.0
    ann = float(sub[-1] ** (1.0 / yrs) - 1) if sub[-1] > 0 and yrs > 0 else -1.0
    sd = rsub.std()
    sharpe = float(rsub.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    mdd = float((sub / np.maximum.accumulate(sub) - 1).min())
    return dict(total=float(sub[-1] - 1), ann=ann, sharpe=sharpe, mdd=mdd, n=int(mask.sum()))


def main():
    from gold_tide_engine import detect_tides
    df = load_data(CONFIG['DATA_PATH'])
    atr_full = compute_atr(df, CONFIG['ATR_WINDOW'])
    feat = pd.read_csv('D:/Work/module/gold_features.csv', parse_dates=['Date'])
    # 对齐：特征行 t 对应原序列行 t+warm
    warm = _warm(df, atr_full)
    close = df['Close'].values.astype(float)[warm:]
    atr = atr_full[warm:]
    sma = pd.Series(close).rolling(250, min_periods=50).mean().values

    prob = gen_p(feat, 10)
    print('=' * 72)
    print('T3 风险叠加：信号(已验证) + 置信/动能/vol/regime/ATR止损')
    print('=' * 72)
    for cost in (0.0005, 0.002):
        r_raw = strategy_equity(close, atr, sma, feat, prob, 'raw', cost)
        r_naked = strategy_equity(close, atr, sma, feat, prob, 'naked', cost)
        r_t3 = strategy_equity(close, atr, sma, feat, prob, 't3', cost)
        keys = sorted(prob.keys())
        bh = close[keys[-1]] / close[keys[0]] - 1
        print('\n[成本=%.2f%%/边]  (对齐warm=%d)' % (cost * 100, warm))
        print('  raw(无止损): 总=%+.0f%% 年化=%+.1f%% 夏普=%.2f 回撤=%+.1f%%' % (
            r_raw['total'] * 100, r_raw['ann'] * 100, r_raw['sharpe'], r_raw['mdd'] * 100))
        print('  裸信号(+止损): 总=%+.0f%% 年化=%+.1f%% 夏普=%.2f 回撤=%+.1f%%' % (
            r_naked['total'] * 100, r_naked['ann'] * 100, r_naked['sharpe'], r_naked['mdd'] * 100))
        print('  T3叠加(+止损): 总=%+.0f%% 年化=%+.1f%% 夏普=%.2f 回撤=%+.1f%%' % (
            r_t3['total'] * 100, r_t3['ann'] * 100, r_t3['sharpe'], r_t3['mdd'] * 100))
        print('  买持有: 总=%+.0f%%' % (bh * 100))


if __name__ == '__main__':
    main()
