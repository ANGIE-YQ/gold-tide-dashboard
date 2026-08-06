"""
gold_tide_en_v.py — ±En·ʋ = En+1 特征化模块
=============================================
将状态转移框架的参数转化为每根bar上的连续特征,
供ML模型学习。不产生稀疏信号。

每根bar输出24维特征(3窗口 × 8参数):
  adv      : 当前状态优势度 [-1,+1]
  abs      : |优势度| (距归零距离)
  v_up     : 上涨趋变系数
  v_down   : 下跌趋变系数
  divergence: |v_up - v_down| (方向分歧度)
  accel    : 优势度变化速度
  up_energy: 滚动窗口上涨总能量
  down_energy: 滚动窗口下跌总能量
"""

import numpy as np
import pandas as pd
from gold_tide_engine import load_data, compute_atr, detect_tides
from gold_tide_score import score_all_tides


def compute_advantage(up_e, down_e):
    up_t = sum(up_e) if up_e else 0
    down_t = sum(down_e) if down_e else 0
    total = up_t + down_t
    return (up_t - down_t) / total if total > 1e-9 else 0.0


def compute_v(energies):
    if len(energies) < 2:
        return 1.0
    ratios = [energies[i] / max(energies[i-1], 1e-9)
              for i in range(1, len(energies))]
    return float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0


def build_en_v_features(feat_df, small_legs, dates, window_sizes=(10, 20, 30)):
    """
    在已有特征DataFrame上追加±En·ʋ特征列。
    
    Args:
        feat_df: 基础特征DataFrame (gold_features.csv)
        small_legs: 小潮汐列表 (带energy字段)
        dates: 原始数据Date数组
        window_sizes: 多时间窗口(小潮汐个数)
    
    Returns:
        追加了en_v_*列的特征DataFrame
    """
    for s in small_legs:
        if 'energy' not in s:
            s['energy'] = s['height'] * max(s['duration_bars'], 1)

    for ws in window_sizes:
        cols = [
            f'en_v_adv_{ws}', f'en_v_abs_{ws}',
            f'en_v_v_up_{ws}', f'en_v_v_down_{ws}',
            f'en_v_divergence_{ws}', f'en_v_accel_{ws}',
            f'en_v_up_energy_{ws}', f'en_v_down_energy_{ws}',
        ]
        for c in cols:
            feat_df[c] = 0.0

        prev_adv = 0.0

        for i in range(ws, len(small_legs)):
            end_idx = small_legs[i]['end_idx']
            feat_rows = feat_df[feat_df['Date'] == dates[end_idx]]
            if len(feat_rows) == 0:
                continue
            feat_idx = feat_rows.index[0]

            win = small_legs[max(0, i-ws):i+1]
            up_e = [s['energy'] for s in win if s['direction'] == 'up']
            down_e = [s['energy'] for s in win if s['direction'] == 'down']

            if len(up_e) < 2 or len(down_e) < 2:
                continue

            adv = compute_advantage(up_e, down_e)
            v_up = compute_v(up_e[-8:])
            v_down = compute_v(down_e[-8:])
            divergence = abs(v_up - v_down) / max(v_up + v_down, 0.01)
            accel = abs(prev_adv) - abs(adv) if abs(prev_adv) > 0 else 0.0

            feat_df.at[feat_idx, f'en_v_adv_{ws}'] = round(adv, 4)
            feat_df.at[feat_idx, f'en_v_abs_{ws}'] = round(abs(adv), 4)
            feat_df.at[feat_idx, f'en_v_v_up_{ws}'] = round(v_up, 3)
            feat_df.at[feat_idx, f'en_v_v_down_{ws}'] = round(v_down, 3)
            feat_df.at[feat_idx, f'en_v_divergence_{ws}'] = round(divergence, 4)
            feat_df.at[feat_idx, f'en_v_accel_{ws}'] = round(accel, 4)
            feat_df.at[feat_idx, f'en_v_up_energy_{ws}'] = round(sum(up_e), 0)
            feat_df.at[feat_idx, f'en_v_down_energy_{ws}'] = round(sum(down_e), 0)

            prev_adv = adv

    # 向前填充(每根bar用最近一次更新)
    for ws in window_sizes:
        for col in [
            f'en_v_adv_{ws}', f'en_v_abs_{ws}',
            f'en_v_v_up_{ws}', f'en_v_v_down_{ws}',
            f'en_v_divergence_{ws}', f'en_v_accel_{ws}',
            f'en_v_up_energy_{ws}', f'en_v_down_energy_{ws}',
        ]:
            feat_df[col] = feat_df[col].replace(0.0, np.nan).ffill().fillna(0.0)

    return feat_df


def main():
    """独立运行: 构建增强特征并保存"""
    import os
    from gold_tide_features import build_features

    BASE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE)

    df = load_data('gold_AU0_daily.csv')
    atr = compute_atr(df, 20)
    big, small, big_pv, small_pv = detect_tides(df, 3.0, 1.0, atr)
    score_all_tides(df, big, small, atr)

    # 基础特征层
    feat = build_features(df, atr, big, small, small_pv, save_csv=False)

    # 追加±En·ʋ特征
    dates_arr = df['Date'].values
    feat = build_en_v_features(feat, small, dates_arr)

    feat.to_csv('gold_features_enhanced.csv', index=False)
    print(f'  特征层: {feat.shape[0]}行 × {feat.shape[1]}列')
    print(f'  ±En·ʋ 特征: {sum(1 for c in feat.columns if c.startswith("en_v_"))}维')
    print(f'  保存: gold_features_enhanced.csv')


if __name__ == '__main__':
    main()
