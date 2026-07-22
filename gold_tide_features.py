"""
gold_tide_features.py —— 无前视条形级特征层（框架价值最大化的核心资产）
把潮汐引擎的输出，重构为「每根 bar 可用、严格无前视」的特征矩阵，供树模型学习。

无前视保证：
  · 当前潮汐用 start_idx<=t 识别（其 end_idx 在将来，不使用）；
  · 所有"运行态"指标只在 [start..t] 上累积；
  · "上一潮汐"是已完成的（end<当前潮汐start<=t），其完整12项动能已知；
  · 拐点只用 conf_idx<=t 的（确认当下才可知）；
  · 标签用 t+1 之后的前向收益。
"""
import numpy as np
import pandas as pd
from gold_tide_engine import load_data, detect_tides, compute_atr, CONFIG
from gold_tide_score import score_all_tides

KEYS = None  # 12 项动能键，build 时填充


def build_features(df, atr, big_legs, small_legs, small_pv, save_csv=True):
    global KEYS
    close = df['Close'].values.astype(float)
    n = len(close)
    starts = np.array([b['start_idx'] for b in big_legs])
    contain = np.searchsorted(starts, np.arange(n), side='right') - 1
    contain = np.clip(contain, 0, len(big_legs) - 1)

    bt_dir = np.array([1 if b['direction'] == 'up' else -1 for b in big_legs])
    bt_mom = np.array([b['momentum_score'] for b in big_legs])
    bt_energy = np.array([b['energy'] for b in big_legs])
    avg_energy = float(np.mean(bt_energy)) or 1.0
    KEYS = sorted(big_legs[0]['momentum_detail'].keys())
    bt_sub = np.array([[b['momentum_detail'][k] for k in KEYS] for b in big_legs])
    # 上一潮汐的"一键"特征
    bt_oneway = np.zeros(len(big_legs)); bt_band = np.zeros(len(big_legs))
    for i, b in enumerate(big_legs):
        seg = close[b['start_idx']:b['end_idx'] + 1]
        net = abs(seg[-1] - seg[0]); path = np.sum(np.abs(np.diff(seg))) + 1e-9
        bt_oneway[i] = min(net / path, 1.0)
        _, resid, bw = _tide_reg_at(b, close)
        bt_band[i] = 1 - min(bw / (b['height'] + 1e-9), 1.0)

    avg_small_energy = float(np.mean([s['energy'] for s in small_legs])) or 1.0
    # 最近确认拐点（conf_idx<=t）
    pv = small_pv[small_pv['confirmed']].sort_values('conf_idx')
    pv_conf = pv['conf_idx'].values.astype(int)
    pv_kind = pv['kind'].values
    pv_idx = pv['idx'].values.astype(int)
    small_energy_of = {int(s['end_idx']): s['energy'] for s in small_legs}  # 小潮汐末端极值→能量
    last_conf = np.full(n, -1, dtype=int)
    last_kind = np.array([''] * n, dtype=object)
    last_e = np.zeros(n)
    j = 0
    for t in range(n):
        while j < len(pv_conf) and pv_conf[j] <= t:
            last_conf[t] = pv_conf[j]; last_kind[t] = pv_kind[j]
            last_e[t] = small_energy_of.get(pv_idx[j], 0.0)
            j += 1
    # 向后填充（沿用最近确认拐点）
    for t in range(1, n):
        if last_conf[t] == -1:
            last_conf[t] = last_conf[t - 1]; last_kind[t] = last_kind[t - 1]; last_e[t] = last_e[t - 1]

    avg_atr = float(np.mean(atr)) or 1.0

    # 特征列
    cols = ['cur_dir', 'cur_bars_in', 'cur_into_norm', 'cur_partial_ret',
            'cur_energy_ratio', 'cur_slope_norm', 'cur_band_norm', 'cur_oneway',
            'prev_mom', 'prev_dir', 'prev_energy_ratio', 'prev_oneway', 'prev_band_norm'] + \
           ['prev_' + k for k in KEYS] + \
           ['lp_L', 'lp_H', 'lp_bars_since', 'lp_energy_ratio', 'atr_norm_ret', 'vol_ratio']
    X = np.zeros((n, len(cols)))

    cur = -2
    Sx = Sxx = Sxy = Sy = Syy = cnt = 0.0
    path_sum = 0.0
    for t in range(n):
        bi = int(contain[t])
        if bi != cur:
            cur = bi
            Sx = Sxx = Sxy = Sy = Syy = cnt = 0.0
            path_sum = 0.0
            started = True
        s = big_legs[bi]['start_idx']
        x = t - s
        y = close[t]
        Sx += x; Sxx += x * x; Sxy += x * y; Sy += y; Syy += y * y; cnt += 1
        if t > 0:
            path_sum += abs(close[t] - close[t - 1])
        d = big_legs[bi]['direction']
        dirn = 1 if d == 'up' else -1
        start_px = close[s]
        partial_h = abs(y - start_px)
        partial_ret = (y - start_px) * dirn / (start_px + 1e-9)
        energy_run = partial_h * max(cnt, 1)
        slope = 0.0; band = 0.0; oneway = 0.0
        if cnt >= 2:
            denom = cnt * Sxx - Sx * Sx
            if denom > 1e-12:
                slope = (cnt * Sxy - Sx * Sy) / denom
            intercept = (Sy - slope * Sx) / cnt if cnt > 0 else 0.0
            resid_sum = Syy - 2 * slope * Sxy - 2 * intercept * Sy + slope * slope * Sxx + \
                        2 * slope * intercept * Sx + cnt * intercept * intercept
            std = np.sqrt(max(resid_sum, 0.0) / (cnt - 1))
            band = 1 - min((2 * std) / (partial_h + 1e-9), 1.0)
            net = partial_h
            oneway = min(net / (path_sum + 1e-9), 1.0)
        slope_norm = slope / (start_px + 1e-9)

        pv_idx = int(contain[t]) - 1
        if pv_idx >= 0:
            pmom = bt_mom[pv_idx]; pdir = bt_dir[pv_idx]
            penergy = bt_energy[pv_idx] / avg_energy
            poneway = bt_oneway[pv_idx]; pband = bt_band[pv_idx]
            psub = bt_sub[pv_idx]
        else:
            pmom = pdir = penergy = poneway = pband = 0.0
            psub = np.zeros(len(KEYS))

        lpL = 1.0 if last_kind[t] == 'L' else 0.0
        lpH = 1.0 if last_kind[t] == 'H' else 0.0
        lp_since = (t - last_conf[t]) if last_conf[t] >= 0 else 9999
        lp_e = last_e[t] / avg_small_energy if avg_small_energy > 0 else 0.0
        anr = ((close[t] - (close[t - 1] if t > 0 else close[t])) / (close[t - 1] if t > 0 else close[t])) / \
              (atr[t] / (close[t] + 1e-9) + 1e-9)
        volr = atr[t] / avg_atr

        row = [dirn, cnt - 1, (cnt - 1) / 26.0, partial_ret, energy_run / avg_energy,
               slope_norm, band, oneway,
               pmom, pdir, penergy, poneway, pband] + list(psub) + \
              [lpL, lpH, lp_since, lp_e, anr, volr]
        X[t] = row

    feat_df = pd.DataFrame(X, columns=cols)
    feat_df['Date'] = df['Date'].values
    feat_df['Close'] = close
    # 标签：t+1 入场，fwd 根后收益
    for k in (5, 10, 20):
        fr = np.full(n, np.nan)
        for t in range(n):
            if t + 1 + k < n:
                fr[t] = close[t + 1 + k] / close[t + 1] - 1.0
        feat_df['fwd%d' % k] = fr
    # 预热：跳过首个大潮汐起点前 + 20 根
    warm = starts[0] + 20
    feat_df = feat_df.iloc[warm:].reset_index(drop=True)
    if save_csv:
        feat_df.to_csv('D:/Work/module/gold_features.csv', index=False)
    return feat_df


def _tide_reg_at(leg, close):
    """复用 score 模块的通道计算（用于已完成潮汐，无前视）。"""
    from gold_tide_score import tide_regression
    return tide_regression(pd.DataFrame({'Close': close}), leg)


def main():
    df = load_data(CONFIG['DATA_PATH'])
    atr = compute_atr(df, CONFIG['ATR_WINDOW'])
    big, small, _, spv = detect_tides(df, 3, 1, atr)
    score_all_tides(df, big, small, atr)
    feat = build_features(df, atr, big, small, spv)
    print('特征矩阵: %d 行 × %d 列' % (feat.shape[0], feat.shape[1]))
    print('保存: D:/Work/module/gold_features.csv')
    print('特征列:', list(feat.columns[:-4]))


if __name__ == '__main__':
    main()
