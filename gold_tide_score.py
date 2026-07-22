"""
gold_tide_score.py
(接续 gold_tide_engine.py 的分段结果)

将你「评估总结」12 条 + 「边界约束法则」量化为可计算指标:
  · 每个大潮汐  -> 综合动能分(0-100) + 12 项分项明细
  · 每个反向节(小潮汐拐点) -> 信号 BUY / SELL / HOLD(边界处建仓)
  · 渐变法则    -> 比较当前与上一同向潮汐动能,预判延续/反转
  · 强约束(第9条) -> 通道带宽窄 = 约束强 = 动能强

信号纪律(对应你的三步定式):
  1) 大潮汐动能分 + 渐变对比,判断动能态势(强/弱)
  2) 动能弱 -> 反向节处预判反转;动能强 -> 顺势延续
  3) 在反向节(边界拐点)发出信号,次日以收盘价交易

无前视偏差:信号在反向节「确认」时(t 时刻, 纯历史信息)发出.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gold_tide_engine import load_data, detect_tides, compute_atr, CONFIG


# ----------------------------- 通道与约束 -----------------------------
def tide_regression_range(df, s, e):
    """对 close[s..e] 做线性回归,返回拟合线,残差,通道带宽(±1sigma)."""
    xs = np.arange(s, e + 1, dtype=float)
    ys = df['Close'].values.astype(float)[s:e + 1]
    A = np.polyfit(xs, ys, 1)
    fit = np.polyval(A, xs)
    resid = ys - fit
    bandwidth = 2.0 * np.std(resid) + 1e-9
    return fit, resid, bandwidth


def tide_regression(df, leg):
    """对潮汐内收盘价做线性回归,返回拟合线,残差,通道带宽(±1sigma)."""
    return tide_regression_range(df, leg['start_idx'], leg['end_idx'])


def touch_count(resid, bandwidth):
    """价格触碰通道边缘(±0.5xbandwidth)的局部极值次数 -> 边界磨损代理."""
    thr = 0.5 * bandwidth
    touch = 0
    for i in range(1, len(resid) - 1):
        if (resid[i] >= thr and resid[i] >= resid[i-1] and resid[i] >= resid[i+1]) or \
           (resid[i] <= -thr and resid[i] <= resid[i-1] and resid[i] <= resid[i+1]):
            touch += 1
    return touch


# ----------------------------- 12 条规则 -> 动能分项 -----------------------------
def _score_one_leg(df, leg, subs_all, atr, avg_big_dur, avg_small_dur, avg_energy, end_eval=None):
    """对单条大潮汐做 12 项动能评分.

    end_eval=None  -> 用完整潮汐(start..end_idx),即原 score_all_tides 口径;
    end_eval=K     -> 运行态评分,只用 [start..K] 的数据与「在 K 之前已结束」的小潮汐,
                     严格无前视(用于信号在潮汐内拐点发出时的『当时已知』动能评估).
    返回 (composite 0-100, sc 字典).
    """
    close = df['Close'].values.astype(float)
    s = int(leg['start_idx'])
    if end_eval is not None:
        # 运行态:评估到拐点确认 bar;子潮汐只取在该 bar 之前已结束的.
        # 注意:不把 e 钳制到 leg['end_idx']----大潮汐末端需 3xATR 回撤才确认,
        # 其内子潮汐(end_idx)可能超出 leg['end_idx'](如末条大潮汐),运行评估应含到 conf_idx.
        e = max(int(end_eval), s)
        subs = [sl for sl in subs_all if sl['parent_big'] == leg['tide_id'] and sl['end_idx'] <= e]
    else:
        # 完整潮汐:含该大潮汐下全部子潮汐(含 end_idx 超出 leg['end_idx'] 者,与原始口径一致)
        e = int(leg['end_idx'])
        subs = [sl for sl in subs_all if sl['parent_big'] == leg['tide_id']]
    seg = close[s:e + 1]
    height = abs(close[e] - close[s]) + 1e-9            # 运行高度
    dur = max(e - s, 1)                                # 运行时长
    energy = height * dur                              # 运行能量(高度x时间)
    sub_e = [sl['energy'] for sl in subs]
    _, resid, bandwidth = tide_regression_range(df, s, e)

    sc = {}
    # 1 边界磨损:同方向反复触及边界无突破 -> 动能衰竭(弱)
    tc = touch_count(resid, bandwidth)
    end_break = 1 if abs(resid[-1]) > 0.5 * bandwidth else 0
    wear = 1.0 if (tc >= 3 and end_break == 0) else min(tc / 6.0, 1.0)
    sc['s1_wear'] = 1 - wear
    # 2 首节即高位 -> 动能强
    if subs:
        fr = abs(subs[0]['end_price'] - subs[0]['start_price']) / height
        sc['s2_first'] = min(fr / 0.5, 1.0)
    else:
        sc['s2_first'] = 0.5
    # 3 一字型(内部折返少) -> 动能强
    internal = max(0, len(subs) - 2)
    sc['s3_straight'] = 1 - min(internal / 4.0, 1.0)
    # 4 节数长 -> 动能强
    sc['s4_dur'] = min(dur / avg_big_dur, 1.0)
    # 5 反向节高位且短促 -> 动能强
    last_dur = subs[-1]['duration_bars'] if subs else dur
    sc['s5_prompt'] = max(0.0, min(1 - last_dur / avg_small_dur, 1.0)) if last_dur < avg_small_dur else 0.0
    # 6 光滑流畅(残差小) -> 动能强
    sc['s6_smooth'] = 1 - min(np.std(resid) / height, 1.0)
    # 7 高度x时间=动能 -> 量纲(用比值,严格尺度不变)
    sc['s7_energy'] = min(energy / (2.0 * avg_energy), 1.0)
    # 8 均匀释放(各小潮汐能量 CV 低) -> 动能强
    if len(sub_e) >= 2 and np.mean(sub_e) > 0:
        cv = np.std(sub_e) / np.mean(sub_e)
        sc['s8_uniform'] = 1 - min(cv, 1.0)
    else:
        sc['s8_uniform'] = 0.5
    # 9 强约束(通道窄) -> 动能强  【你确认:强弱束 = 强约束】
    sc['s9_constraint'] = 1 - min(bandwidth / height, 1.0)
    # 10 先期强 -> 动能强;后期强 -> 弱
    if len(sub_e) >= 2:
        mid = len(sub_e) // 2
        front = np.sum(sub_e[:mid + 1]); back = np.sum(sub_e[mid:])
        sc['s10_front'] = min(front / (back + 1e-9), 1.0)
    else:
        sc['s10_front'] = 0.5
    # 11 单边(直线度高) -> 动能强;双边(曲折) -> 弱
    total_path = np.sum(np.abs(np.diff(seg))) + 1e-9
    net = abs(seg[-1] - seg[0])
    sc['s11_oneway'] = min(net / total_path, 1.0)
    # 12 反转突变(单位时间反转幅度大) -> 动能强
    steep = height / dur
    avg_steep = avg_energy / avg_big_dur
    sc['s12_reversal'] = min(steep / (avg_steep + 1e-9), 1.0)

    composite = float(np.mean(list(sc.values()))) * 100.0
    return composite, sc


def score_all_tides(df, big_legs, small_legs, atr):
    """对每条大潮汐(完整口径)计算 12 项动能分 + 渐变对比(上一同向动能分)."""
    close = df['Close'].values.astype(float)
    avg_big_dur = float(np.mean([l['duration_bars'] for l in big_legs])) or 1.0
    avg_small_dur = float(np.mean([l['duration_bars'] for l in small_legs])) or 1.0
    avg_energy = float(np.mean([l['energy'] for l in big_legs])) or 1.0
    last_up = None
    last_down = None

    for leg in big_legs:
        composite, sc = _score_one_leg(df, leg, small_legs, atr,
                                       avg_big_dur, avg_small_dur, avg_energy, end_eval=None)
        leg['momentum_score'] = round(composite, 1)
        leg['momentum_detail'] = sc
        # 渐变法则:记录上一同向潮汐动能分
        if leg['direction'] == 'up':
            leg['prev_same_dir_score'] = last_up
            last_up = composite
        else:
            leg['prev_same_dir_score'] = last_down
            last_down = composite
    return big_legs


def pivot_momentum(df, big_legs, small_legs, atr, owner_leg, conf_idx):
    """运行动能(无前视):在拐点被确认的 bar(conf_idx) 处,对所属大潮汐做
    『当时已知』的动能评分,仅用 <=conf_idx 的数据.信号在潮汐内拐点发出时,
    不再偷看潮汐后续(约 2 根)的未来数据.返回 0-100 分值."""
    avg_big_dur = float(np.mean([l['duration_bars'] for l in big_legs])) or 1.0
    avg_small_dur = float(np.mean([l['duration_bars'] for l in small_legs])) or 1.0
    avg_energy = float(np.mean([l['energy'] for l in big_legs])) or 1.0
    composite, _ = _score_one_leg(df, owner_leg, small_legs, atr,
                                  avg_big_dur, avg_small_dur, avg_energy, end_eval=int(conf_idx))
    return composite


# ----------------------------- 信号生成 -----------------------------
def generate_signals(df, big_legs, small_pv, mid=50.0, atr=None, small_legs=None):
    big_sorted = sorted(big_legs, key=lambda x: x['start_idx'])
    recs = []
    for _, p in small_pv[small_pv['confirmed']].iterrows():
        idx = int(p['idx']); conf_idx = int(p['conf_idx']); kind = p['kind']; price = float(p['price'])
        owner = None
        for bl in big_sorted:
            if bl['start_idx'] <= idx <= bl['end_idx']:
                owner = bl
                break
        if owner is None:
            cand = [b for b in big_sorted if b['start_idx'] <= idx]
            owner = cand[-1] if cand else big_sorted[-1]
        # 运行动能(无前视):用拐点确认 bar 处『当时已知』的动能分,而非完整潮汐分
        # (完整潮汐分用到潮汐后续约2根数据,属微泄漏;此处修复)
        if atr is not None and small_legs is not None:
            score = pivot_momentum(df, big_legs, small_legs, atr, owner, conf_idx)
        else:
            score = owner.get('momentum_score', 50.0)   # 兼容旧调用(含约2根微泄漏)
        prev = owner.get('prev_same_dir_score')
        # 渐变:当前动能 < 上一同向动能 -> 该方向动能渐弱 -> 预判反转
        grad_weak = (prev is not None and score < prev) or (score < mid)
        if kind == 'H':   # 上行潮汐结束(高点) -> 弱则翻转做空
            sig = 'SELL' if grad_weak else 'HOLD'
        else:             # 下行潮汐结束(低点) -> 弱则翻转做多
            sig = 'BUY' if grad_weak else 'HOLD'
        recs.append({'idx': idx, 'conf_idx': conf_idx, 'date': df['Date'].values[idx], 'kind': kind,
                     'price': price, 'big': owner['tide_id'],
                     'score': score, 'signal': sig})
    return pd.DataFrame(recs)


# ----------------------------- 回测雏形 -----------------------------
def backtest(df, signals, fwd=10):
    """
    诚实的方向预测检验(已修正前视偏差).
    信号在拐点被「确认」的 bar(conf_idx) 才可知;真实可执行入场 = conf_idx+1(次日收盘价).
    此前版本在极值 bar(idx)+1 入场,等于偷看了未来的确认信息,并被 ZigZag 确认机制
    机械抬升命中率(确认要求价格从极值回撤 thr,故 idx+1 入场天然领先 thr).
    平仓用固定持有期 fwd,不依赖下一个 ZigZag 拐点,真正检验方向预测力.
    返回 (有效信号数, 方向命中数, 方向准确率%, 累计收益点).
    """
    close = df['Close'].values.astype(float)
    n = len(close)
    hit = 0
    total_ret = 0.0
    cnt = 0
    for _, r in signals[signals['signal'] != 'HOLD'].iterrows():
        i = int(r['conf_idx'])  # 拐点确认 bar(此前错误地用了极值 idx)
        if i + 1 + fwd >= n:
            continue
        entry = close[i + 1]
        exit = close[i + 1 + fwd]
        ret = (exit - entry) if r['signal'] == 'BUY' else (entry - exit)
        if ret > 0:
            hit += 1
        total_ret += ret
        cnt += 1
    acc = 100.0 * hit / max(cnt, 1)
    return cnt, hit, acc, total_ret


# ----------------------------- 可视化 -----------------------------
def plot_signals(df, big_legs, signals, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(df['Date'], df['Close'], color='black', lw=0.7)
    cmap = {'up': '#1f77b4', 'down': '#d62728'}
    for leg in big_legs:
        xs = df['Date'].values[leg['start_idx']:leg['end_idx']+1]
        ys = df['Close'].values[leg['start_idx']:leg['end_idx']+1]
        ax1.plot(xs, ys, color=cmap[leg['direction']], lw=1.6, alpha=0.7)
    buy = signals[signals['signal'] == 'BUY']
    sell = signals[signals['signal'] == 'SELL']
    ax1.scatter(df['Date'].values[buy['idx']], buy['price'], marker='^', s=70,
                color='green', zorder=6, label='BUY')
    ax1.scatter(df['Date'].values[sell['idx']], sell['price'], marker='v', s=70,
                color='red', zorder=6, label='SELL')
    ax1.set_title('Gold Tide -- Momentum Score & Trading Signals')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.25)
    dates = [bl['start_date'] for bl in big_legs]
    scores = [bl['momentum_score'] for bl in big_legs]
    ax2.plot(dates, scores, color='purple', lw=1.2)
    ax2.axhline(50, color='gray', ls='--', lw=0.8)
    ax2.set_ylabel('Momentum(0-100)')
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------- 主流程 -----------------------------
def main():
    cfg = CONFIG
    df = load_data(cfg['DATA_PATH'])
    atr = compute_atr(df, cfg['ATR_WINDOW'])
    big_legs, small_legs, big_pv, small_pv = detect_tides(
        df, cfg['BIG_THR_MULT'], cfg['SMALL_THR_MULT'], atr)
    score_all_tides(df, big_legs, small_legs, atr)
    sig = generate_signals(df, big_legs, small_pv)
    n_sig, hit, acc, cum = backtest(df, sig, fwd=10)
    plot_signals(df, big_legs, sig, 'D:/Work/module/gold_signals.png')

    scores = [b['momentum_score'] for b in big_legs]
    print('=' * 64)
    print('动能评分分布: min=%.1f  median=%.1f  mean=%.1f  max=%.1f' % (
        min(scores), float(np.median(scores)), float(np.mean(scores)), max(scores)))
    print('=' * 64)
    print('\n最近大潮汐动能评分(含渐变对比):')
    rows = []
    for b in big_legs[-14:]:
        rows.append({
            'BigTide': b['tide_id'], 'Dir': b['direction'],
            'Score': b['momentum_score'],
            'PrevSameDir': ('' if b['prev_same_dir_score'] is None
                            else round(b['prev_same_dir_score'], 1)),
            '渐变': ('--' if b['prev_same_dir_score'] is None
                     else ('弱↓' if b['momentum_score'] < b['prev_same_dir_score'] else '强↑')),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    best = max(big_legs, key=lambda x: x['momentum_score'])
    print('\n示例潮汐 %s (Score=%.1f, %s) 的12项动能明细:' % (
        best['tide_id'], best['momentum_score'], best['direction']))
    for k, v in best['momentum_detail'].items():
        print('  %s = %.2f' % (k, v))

    print('\n信号统计: BUY=%d  SELL=%d  HOLD=%d' % (
        int((sig['signal'] == 'BUY').sum()), int((sig['signal'] == 'SELL').sum()),
        int((sig['signal'] == 'HOLD').sum())))
    print('\n最近信号(反向节):')
    print(sig.tail(12)[['date', 'kind', 'price', 'big', 'score', 'signal']].to_string(index=False))
    print('\n方向预测检验(信号次日进场, 固定持有10根平仓, 平仓独立于分段):')
    print('  有效信号 = %d   方向命中 = %d   方向准确率 = %.1f%%' % (n_sig, hit, acc))
    print('  累计收益(点, 未计成本) = %.1f   单笔均值 = %.2f' % (cum, cum / max(n_sig, 1)))
    print('  注: 原始「拐点端点」回测胜率会虚高(≈100%),因买卖点=分段端点,')
    print('      属循环论证,不能证明预测力;以上为诚实的方向预测口径.')
    print('\n图表: D:/Work/module/gold_signals.png')


if __name__ == '__main__':
    main()
