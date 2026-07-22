"""
gold_tide_engine.py

【核心映射 / 操作化定义 ---- 请用户确认是否符合原意】
  · 大潮汐  = 以「粗阈值」ZigZag 切出的方向性波段(腿),编号 A, B, C, D, E ...
  · 小潮汐  = 以「细阈值」ZigZag 切出的方向性波段(腿),在其所属大潮汐内编号 a, b, c, d ...
             命名形态即「双名法」:如 Aa, Ab(大A的第a/b小潮汐)
  · 反向节  = 小潮汐的端点(拐点),即操作节点
  · 潮汐原点 = 大潮汐起点
  · 动能近似 = 高度 x 时间(你第7条:高度x时间表述为潮汐动能)

【无前视偏差保证 ---- 建模模底线】
  · ZigZag 仅在价格从极值回撤超过「动态阈值」时才确认该极值为拐点;
    拐点位置 = 历史极值所在 bar(截至当时已发生),不使用任何未来信息.
  · 动态阈值 = mult x ATR(n),ATR 用滚动窗口(纯历史)计算,无未来函数.
  · 序列末端的极值尚未被回撤确认,标 confirmed=False,回测时剔除.

【数据源】
  · 默认读取本地已落地 CSV(沪金主力 AU0 日线,真实数据 2008-2026).
  · 引擎同时提供 fetch_sina() / fetch_yfinance(),可一键切换到美元金价(COMEX/GC=F).
  · 美元金价只需把 DATA_PATH 指向 GC=F 的 CSV,后续动能评分/信号逻辑完全一致.
"""

import matplotlib
matplotlib.use('Agg')  # 无显示环境,必须
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ============================ CONFIG(假设独立区) ============================
CONFIG = {
    'DATA_PATH': 'D:/Work/module/gold_AU0_daily.csv',  # 已落地的真实数据
    'ATR_WINDOW': 20,        # ATR 滚动窗口(交易日)
    'BIG_THR_MULT': 3.0,     # 大潮汐阈值 = 3 x ATR(粗)
    'SMALL_THR_MULT': 1.0,   # 小潮汐阈值 = 1 x ATR(细)
    'FIG_PATH': 'D:/Work/module/gold_tides.png',
    'N_RECENT': 14,          # 摘要展示最近 N 个大潮汐
}
# ============================================================================


# ----------------------------- 数据层 -----------------------------
def load_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def fetch_sina(symbol='AU0', save_to=None):
    """从新浪拉沪金主力日线(真实数据,人民币/克).可保存为 CSV 复用."""
    import urllib.request, json
    url = (f'https://stock2.finance.sina.com.cn/futures/api/json.php/'
           f'InnerFuturesNewService.getDailyKLine?symbol={symbol}')
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
                                 timeout=25).read().decode('gbk', 'ignore')
    d = json.loads(raw)
    df = pd.DataFrame(d).rename(columns={'d': 'Date', 'o': 'Open', 'h': 'High',
                                         'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Close']).sort_values('Date').reset_index(drop=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    if save_to:
        df.to_csv(save_to, index=False)
    return df


def fetch_yfinance(symbol='GC=F', start='2018-01-01', end='2026-07-20'):
    """切换到美元金价:COMEX 黄金期货连续(GC=F) 或 黄金ETF(GLD).需本地可联网."""
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


# ----------------------------- 指标层 -----------------------------
def compute_atr(df, n):
    """Wilder 式 ATR(滚动均值,纯历史,无未来函数)."""
    high, low, close = df['High'].values, df['Low'].values, df['Close'].values
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


# ----------------------------- 核心:纯前向 ZigZag -----------------------------
def zigzag(df, mult, atr):
    """
    纯前向拐点检测.
    仅在价格从当前极值回撤 >= multxATR 时,确认该极值所在 bar 为拐点.
    返回 DataFrame: idx, price, kind(H/L), confirmed.
    最后一个极值为未确认(provisional),回测时剔除.
    """
    close = df['Close'].values
    n = len(close)
    cand_hi, cand_hi_i = close[0], 0
    cand_lo, cand_lo_i = close[0], 0
    dirn = None  # None=未定, 1=找高(上一段为低), -1=找低
    pivots = []
    for i in range(1, n):
        thr = mult * atr[i]
        if close[i] > cand_hi:
            cand_hi, cand_hi_i = close[i], i
        if close[i] < cand_lo:
            cand_lo, cand_lo_i = close[i], i
        if dirn is None:
            # 哪边先被反向突破阈值,就确认另一极为拐点
            if cand_hi - close[i] >= thr and cand_hi_i < cand_lo_i:
                pivots.append({'idx': cand_hi_i, 'conf_idx': i, 'price': cand_hi, 'kind': 'H', 'confirmed': True})
                dirn = -1
                cand_lo, cand_lo_i = close[i], i
            elif close[i] - cand_lo >= thr and cand_lo_i < cand_hi_i:
                pivots.append({'idx': cand_lo_i, 'conf_idx': i, 'price': cand_lo, 'kind': 'L', 'confirmed': True})
                dirn = 1
                cand_hi, cand_hi_i = close[i], i
        elif dirn == 1:  # 找高
            if cand_hi - close[i] >= thr:
                pivots.append({'idx': cand_hi_i, 'conf_idx': i, 'price': cand_hi, 'kind': 'H', 'confirmed': True})
                dirn = -1
                cand_lo, cand_lo_i = close[i], i
        else:  # 找低
            if close[i] - cand_lo >= thr:
                pivots.append({'idx': cand_lo_i, 'conf_idx': i, 'price': cand_lo, 'kind': 'L', 'confirmed': True})
                dirn = 1
                cand_hi, cand_hi_i = close[i], i
    # 末端极值未确认
    if dirn == 1:
        pivots.append({'idx': cand_hi_i, 'price': cand_hi, 'kind': 'H', 'confirmed': False})
    else:
        pivots.append({'idx': cand_lo_i, 'price': cand_lo, 'kind': 'L', 'confirmed': False})
    return pd.DataFrame(pivots)


def build_legs(pivots_df):
    """把已确认拐点(交替 H/L)连接成方向性腿(潮汐)."""
    pv = pivots_df[pivots_df['confirmed']].reset_index(drop=True)
    legs = []
    for k in range(len(pv) - 1):
        a, b = pv.iloc[k], pv.iloc[k + 1]
        direction = 'up' if (a['kind'] == 'L' and b['kind'] == 'H') else 'down'
        legs.append({
            'start_idx': int(a['idx']), 'end_idx': int(b['idx']),
            'start_price': float(a['price']), 'end_price': float(b['price']),
            'start_kind': a['kind'], 'end_kind': b['kind'],
            'direction': direction,
        })
    return legs


# ----------------------------- 潮汐识别 + 命名 + 动能预计算 -----------------------------
def detect_tides(df, big_mult, small_mult, atr):
    big_pv = zigzag(df, big_mult, atr)
    small_pv = zigzag(df, small_mult, atr)
    big_legs = build_legs(big_pv)
    small_legs = build_legs(small_pv)
    # 附上每个大潮汐末端拐点的「确认 bar」(conf_idx),供无前视标签锚定使用
    conf_map = {}
    for _, p in big_pv.iterrows():
        if bool(p['confirmed']) and not pd.isna(p.get('conf_idx')):
            conf_map[int(p['idx'])] = int(p['conf_idx'])
    for leg in big_legs:
        leg['end_conf_idx'] = conf_map.get(leg['end_idx'], leg['end_idx'])
    dates = df['Date'].values

    # ---- 大潮汐命名 A, B, C ...
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, leg in enumerate(big_legs):
        leg['tide_id'] = alpha[i] if i < 26 else f'{alpha[i % 26]}{i // 26}'
        leg['start_date'] = dates[leg['start_idx']]
        leg['end_date'] = dates[leg['end_idx']]
        leg['height'] = abs(leg['end_price'] - leg['start_price'])
        leg['duration_bars'] = leg['end_idx'] - leg['start_idx']
        leg['energy'] = leg['height'] * leg['duration_bars']  # 高度x时间≈动能

    # ---- 小潮汐:命名 + 按时间归属大潮汐 ----
    for s in small_legs:
        s['start_date'] = dates[s['start_idx']]
        s['end_date'] = dates[s['end_idx']]
        s['height'] = abs(s['end_price'] - s['start_price'])
        s['duration_bars'] = s['end_idx'] - s['start_idx']
        s['energy'] = s['height'] * s['duration_bars']
        owner = None
        for bl in big_legs:
            if bl['start_idx'] <= s['start_idx'] <= bl['end_idx']:
                owner = bl['tide_id']
                break
        if owner is None:  # fallback:归到起点之前最近的大潮汐
            cand = [bl for bl in big_legs if bl['start_idx'] <= s['start_idx']]
            owner = cand[-1]['tide_id'] if cand else big_legs[0]['tide_id']
        s['parent_big'] = owner

    grp = defaultdict(list)
    for s in small_legs:
        grp[s['parent_big']].append(s)
    for pid, lst in grp.items():
        for j, s in enumerate(lst):
            s['tide_id'] = pid + (chr(ord('a') + j) if j < 26 else f'a{j}')

    cnt = defaultdict(int)
    for s in small_legs:
        cnt[s['parent_big']] += 1
    for bl in big_legs:
        bl['num_subtides'] = cnt.get(bl['tide_id'], 0)

    return big_legs, small_legs, big_pv, small_pv


# ----------------------------- 可视化 -----------------------------
def plot_tides(df, big_legs, small_pv, path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(df['Date'], df['Close'], color='black', lw=0.7, label='Close')
    cmap = {'up': '#1f77b4', 'down': '#d62728'}
    for leg in big_legs:
        xs = df['Date'].values[leg['start_idx']:leg['end_idx'] + 1]
        ys = df['Close'].values[leg['start_idx']:leg['end_idx'] + 1]
        ax.plot(xs, ys, color=cmap[leg['direction']], lw=2.4, alpha=0.85)
        midx = (leg['start_idx'] + leg['end_idx']) // 2
        ax.annotate(leg['tide_id'], (df['Date'].values[midx], df['Close'].values[midx]),
                    fontsize=11, fontweight='bold', color=cmap[leg['direction']],
                    ha='center', va='bottom')
    sp = small_pv[small_pv['confirmed']]
    ax.scatter(df['Date'].values[sp['idx']], sp['price'], marker='o', s=20,
               color='#2ca02c', zorder=5, label='Reversal pivot (sub-tide turn)')
    ax.set_title('Gold Tide Segmentation  (Big Tides A/B/C + Sub-tides a/b/c + Reversal Pivots)')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------- 摘要 -----------------------------
def summarize(big_legs, n):
    rows = []
    for leg in big_legs[-n:]:
        rows.append({
            'BigTide': leg['tide_id'],
            'Dir': leg['direction'],
            'Start': pd.Timestamp(leg['start_date']).date(),
            'End': pd.Timestamp(leg['end_date']).date(),
            'Height': round(leg['height'], 1),
            'Bars': leg['duration_bars'],
            'SubTides': leg['num_subtides'],
            'Energy~H*T': int(leg['energy']),
        })
    return pd.DataFrame(rows)


# ----------------------------- 主流程 -----------------------------
def main():
    cfg = CONFIG
    df = load_data(cfg['DATA_PATH'])
    atr = compute_atr(df, cfg['ATR_WINDOW'])
    big_legs, small_legs, big_pv, small_pv = detect_tides(
        df, cfg['BIG_THR_MULT'], cfg['SMALL_THR_MULT'], atr)
    plot_tides(df, big_legs, small_pv, cfg['FIG_PATH'])

    print(f"数据区间: {df['Date'].min().date()} ~ {df['Date'].max().date()}  共 {len(df)} 根日线")
    print(f"大潮汐数: {len(big_legs)}   小潮汐数: {len(small_legs)}")
    print(f"平均大潮汐时长: {np.mean([l['duration_bars'] for l in big_legs]):.0f} 根 | "
          f"平均小潮汐时长: {np.mean([l['duration_bars'] for l in small_legs]):.0f} 根")
    print(f"阈值: BIG={cfg['BIG_THR_MULT']}xATR  SMALL={cfg['SMALL_THR_MULT']}xATR  ATR窗口={cfg['ATR_WINDOW']}")
    print("\n最近大潮汐摘要(双名法 + 动能预计算字段):")
    print(summarize(big_legs, cfg['N_RECENT']).to_string(index=False))
    print(f"\n图表已保存: {cfg['FIG_PATH']}")


if __name__ == '__main__':
    main()
