"""
gold_tide_optimized.py —— 优化版信号引擎
============================================================
修复:
1. 进行中段特征——不再错误地延续已结束潮汐的方向
2. 双模型集成——Walk-Forward + 全量训练融合
3. 市场状态标签——牛熊/波动率/通道位置

输出: 增强版交易信号 + 置信度分解 + 状态评估
============================================================
"""
import numpy as np
import pandas as pd
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'gold_AU0_daily.csv')
FEAT_PATH = os.path.join(BASE_DIR, 'gold_features.csv')

def build_enhanced_features(force=False):
    """构建增强特征：在原31维基础上追加进行中段修正特征"""
    if os.path.exists(os.path.join(BASE_DIR, 'gold_features_enhanced.csv')) and not force:
        return pd.read_csv(os.path.join(BASE_DIR, 'gold_features_enhanced.csv'), parse_dates=['Date'])

    print('[优化] 构建增强特征层...')

    from gold_tide_engine import load_data, detect_tides, compute_atr
    from gold_tide_score import score_all_tides
    from gold_tide_features import build_features

    df = load_data(DATA_PATH)
    atr = compute_atr(df, 20)
    big, small, big_pv, small_pv = detect_tides(df, 3.0, 1.0, atr)
    score_all_tides(df, big, small, atr)

    # 构建基础特征层
    base_feat = build_features(df, atr, big, small, small_pv, save_csv=False)

    close = df['Close'].values.astype(float)
    n_feat = len(base_feat)
    warm = int(big[0]['start_idx']) + 20

    # ---- 新增特征 ----

    # 1. 进行中段修正：bars after last confirmed tide end
    last_tide_end = big[-1]['end_idx']
    ip_days = np.zeros(n_feat)
    ip_ret = np.zeros(n_feat)
    ip_pos = np.zeros(n_feat)
    ip_small_up_ratio = np.zeros(n_feat)  # 进行中段上涨小潮汐占比
    ip_small_cnt = np.zeros(n_feat)
    ip_last_small_dir = np.zeros(n_feat)
    ip_energy_trend = np.zeros(n_feat)  # 小潮汐能量趋势
    ip_trend_slope = np.zeros(n_feat)

    # 市场状态
    regime = np.zeros(n_feat)  # 1=bull, 0=bear
    vol_regime = np.zeros(n_feat)  # 波动率分位数
    price_vs_ma50 = np.zeros(n_feat)
    price_vs_ma200 = np.zeros(n_feat)
    rsi_14 = np.zeros(n_feat)

    for i in range(n_feat):
        orig_i = i + warm  # 映射回原始序列
        if orig_i >= len(close):
            break

        # 进行中段特征
        if orig_i > last_tide_end:
            days = orig_i - last_tide_end
            ip_days[i] = min(days / 60, 2.0)  # 最多2倍标准化
            ip_ret[i] = close[orig_i] / close[last_tide_end] - 1

            seg = close[last_tide_end:orig_i + 1]
            seg_range = seg.max() - seg.min()
            if seg_range > 0:
                ip_pos[i] = (close[orig_i] - seg.min()) / seg_range

            # 小潮汐统计
            ip_st = [s for s in small if s['start_idx'] >= last_tide_end and s['end_idx'] <= orig_i]
            ip_small_cnt[i] = min(len(ip_st) / 10, 1.0)
            if ip_st:
                up_count = sum(1 for s in ip_st if s['direction'] == 'up')
                ip_small_up_ratio[i] = up_count / len(ip_st)
                ip_last_small_dir[i] = 1 if ip_st[-1]['direction'] == 'up' else -1

                # 能量趋势：最近2个同向小潮汐的能量比
                if len(ip_st) >= 3:
                    recent3 = ip_st[-3:]
                    energies = [s['energy'] for s in recent3]
                    if len(energies) >= 2:
                        slope = np.polyfit(range(len(energies)), energies, 1)[0]
                        ip_energy_trend[i] = np.clip(slope / (np.mean(energies) + 1e-9), -1, 1)

            # 趋势斜率
            if days >= 5:
                ys = close[last_tide_end:orig_i + 1]
                xs = np.arange(len(ys), dtype=float)
                slope_val = np.polyfit(xs, ys, 1)[0]
                ip_trend_slope[i] = slope_val / close[last_tide_end]

        # 市场状态
        if orig_i >= 250:
            sma250 = np.mean(close[orig_i-249:orig_i+1])
            regime[i] = 1.0 if close[orig_i] > sma250 else 0.0

        if orig_i >= 100:
            vol_hist = atr[orig_i-99:orig_i+1] / close[orig_i-99:orig_i+1]
            vol_regime[i] = (atr[orig_i] / close[orig_i]) / (np.percentile(vol_hist, 80) + 1e-9)
            vol_regime[i] = np.clip(vol_regime[i], 0.3, 3.0)

        if orig_i >= 50:
            price_vs_ma50[i] = close[orig_i] / np.mean(close[orig_i-49:orig_i+1]) - 1

        if orig_i >= 200:
            price_vs_ma200[i] = close[orig_i] / np.mean(close[orig_i-199:orig_i+1]) - 1

        # RSI 14
        if orig_i >= 15:
            deltas = np.diff(close[orig_i-14:orig_i+1])
            gain = np.sum(deltas[deltas > 0]) if np.any(deltas > 0) else 0
            loss = -np.sum(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
            rs = gain / loss
            rsi_14[i] = 100 - 100 / (1 + rs)

    # 合并到基础特征
    new_cols = {
        'ip_days': ip_days,
        'ip_ret': ip_ret,
        'ip_pos': ip_pos,
        'ip_small_cnt': ip_small_cnt,
        'ip_small_up_ratio': ip_small_up_ratio,
        'ip_last_small_dir': ip_last_small_dir,
        'ip_energy_trend': ip_energy_trend,
        'ip_trend_slope': ip_trend_slope,
        'regime_bull': regime,
        'vol_regime': vol_regime,
        'price_vs_ma50': price_vs_ma50,
        'price_vs_ma200': price_vs_ma200,
        'rsi_14': rsi_14,
    }
    for col_name, col_data in new_cols.items():
        base_feat[col_name] = col_data

    outpath = os.path.join(BASE_DIR, 'gold_features_enhanced.csv')
    base_feat.to_csv(outpath, index=False)
    print(f'  增强特征: {base_feat.shape[0]}行 x {base_feat.shape[1]}列 -> {outpath}')
    return base_feat


def ensemble_predict(feat, fwd=10):
    """集成预测：优先使用校准模型，融合WF扩展"""
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from gold_tide_t3 import gen_p
    import pickle, os

    model_path = os.path.join(BASE_DIR, 'gold_tide_calibrated_model.pkl')
    dead_feats = {'ip_small_cnt','ip_small_up_ratio','ip_last_small_dir','ip_trend_slope'}
    skip_cols = {'Date', 'Close', 'fwd5', 'fwd10', 'fwd20'}
    fc = [c for c in feat.columns if c not in skip_cols and c not in dead_feats]
    
    y = (feat['fwd%d' % fwd] > 0).astype(int).values
    ok = ~feat['fwd%d' % fwd].isna().values
    X = feat[fc].values
    n_feat = len(fc)

    # 尝试加载已有校准模型
    use_cal = False
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            if saved.get('features') == fc:
                cal_full = saved['model']
                use_cal = True
        except:
            pass

    if not use_cal:
        # 构建校准模型
        base = XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
                              subsample=0.8,colsample_bytree=0.8,
                              eval_metric='logloss',random_state=42)
        base.fit(X[ok], y[ok])
        cal_full = CalibratedClassifierCV(base, method='sigmoid', cv=5)
        cal_full.fit(X[ok], y[ok])
        with open(model_path, 'wb') as f:
            pickle.dump({'model': cal_full, 'features': fc, 'dead': list(dead_feats)}, f)

    # 1. Walk-Forward 概率
    wf_prob = gen_p(feat, fwd)
    wf_last = wf_prob.get(len(feat) - 1, None)

    # 2. 扩展WF（未覆盖时追加）
    wf_ext_p = None
    if wf_last is None:
        valid_idx = np.where(ok)[0]
        if len(valid_idx) > 500:
            train_end = valid_idx[-100]
            if train_end < len(valid_idx):
                X_train = X[valid_idx[:train_end]]
                y_train = y[valid_idx[:train_end]]
                wf_ext_m = XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
                                          subsample=0.8,colsample_bytree=0.8,
                                          eval_metric='logloss',random_state=42)
                wf_ext_m.fit(X_train, y_train)
                wf_ext_p = float(wf_ext_m.predict_proba(X[-1:].reshape(1,-1))[0,1])

    # 3. 校准模型概率
    cur_X = X[-1:].reshape(1,-1)
    full_p = float(cal_full.predict_proba(cur_X)[0,1])

    # 4. 融合
    if wf_last is not None:
        p_final = wf_last*0.6 + full_p*0.4
        method = f'WFx0.6+Calx0.4 (WF={wf_last:.3f},Cal={full_p:.3f})'
    elif wf_ext_p is not None:
        p_final = wf_ext_p*0.5 + full_p*0.5
        method = f'WFextx0.5+Calx0.5 (Ext={wf_ext_p:.3f},Cal={full_p:.3f})'
    else:
        p_final = full_p
        method = f'Calibrated ({full_p:.3f})'

    # 5. 动态仓位：基于置信度+波动率
    conf = abs(p_final - 0.5)
    vol_adj = float(feat.iloc[-1].get('vol_regime', 1.0))
    vol_adj = min(vol_adj, 1.5)  # 高波动不超1.5倍
    
    if conf > 0.15:
        base_pos = 0.20
    elif conf > 0.08:
        base_pos = 0.12
    else:
        base_pos = 0.05
    
    adj_pos = base_pos / vol_adj  # 高波动降仓,低波动加仓
    adj_pos = np.clip(adj_pos, 0.03, 0.25)
    position_pct = f'{adj_pos*100:.0f}%'

    # 特征重要性（从校准模型的基模型获取）
    try:
        base_est = cal_full.estimator if hasattr(cal_full, 'estimator') else cal_full.calibrated_classifiers_[0].base_estimator
    except:
        base_est = cal_full
    imp = sorted(zip(fc, base_est.feature_importances_), key=lambda x: -x[1])[:8]

    new_cols = ['ip_days', 'ip_ret', 'ip_pos', 'ip_small_cnt', 'ip_small_up_ratio',
                'ip_last_small_dir', 'ip_energy_trend', 'ip_trend_slope',
                'regime_bull', 'vol_regime', 'price_vs_ma50', 'price_vs_ma200', 'rsi_14']
    new_imp = {k: v for k, v in imp if k in new_cols}

    return {
        'p_up': p_final,
        'wf_p': wf_last,
        'wf_ext_p': wf_ext_p,
        'full_p': full_p,
        'method': method,
        'confidence': conf,
        'direction': 'BUY' if p_final > 0.55 else ('SELL' if p_final < 0.45 else 'HOLD'),
        'position_pct': position_pct,
        'top_features': imp,
        'new_feature_importance': sum(new_imp.values()),
        'new_features_detail': new_imp,
    }


def market_state(feat):
    """评估当前市场状态"""
    last = feat.iloc[-1]
    states = []

    # 牛熊
    if last.get('regime_bull', 0.5) > 0.5:
        states.append(('牛市', '价格高于250日均线'))
    else:
        states.append(('熊市/调整', '价格低于250日均线'))

    # 波动率
    vr = last.get('vol_regime', 1.0)
    if vr > 1.5:
        states.append(('高波动', f'波动率={vr:.1f}×常态'))
    elif vr < 0.7:
        states.append(('低波动', f'波动率={vr:.1f}×常态'))
    else:
        states.append(('正常波动', f'波动率={vr:.1f}×常态'))

    # 价格位置
    ma50 = last.get('price_vs_ma50', 0)
    if ma50 > 0.03:
        states.append(('超买', f'高于50日均线{ma50*100:.1f}%'))
    elif ma50 < -0.03:
        states.append(('超卖', f'低于50日均线{ma50*100:.1f}%'))

    # RSI
    rsi = last.get('rsi_14', 50)
    if rsi > 70:
        states.append(('RSI超买', f'RSI={rsi:.0f}'))
    elif rsi < 30:
        states.append(('RSI超卖', f'RSI={rsi:.0f}'))

    # 进行中段
    ip_ret = last.get('ip_ret', 0)
    if abs(ip_ret) > 0.05:
        direction = '跌' if ip_ret < 0 else '涨'
        states.append((f'进行中段{direction}', f'{ip_ret*100:+.1f}%'))

    # 小潮汐能量趋势
    et = last.get('ip_energy_trend', 0)
    if et < -0.3:
        states.append(('能量衰减', '小潮汐能量递减→可能反转'))
    elif et > 0.3:
        states.append(('能量增强', '小潮汐能量递增→趋势延续'))

    return states


def print_signal(signal, states, ml_results=None):
    """打印增强版信号简报"""
    p = signal['p_up']
    conf = signal['confidence']

    bar_len = 30
    if p >= 0.5:
        filled = int(p * bar_len)
        bar = '█' * filled + '░' * (bar_len - filled)
    else:
        filled = int((1-p) * bar_len)
        bar = '░' * (bar_len - filled) + '█' * filled

    print()
    print('=' * 68)
    print('  黄金潮汐模型 v2 — 增强信号')
    print('=' * 68)
    print(f'''
  日期: {signal['date']}    价格: {signal['price']:.2f}

  P(涨) = {p:.3f}  [{bar}]  P(跌) = {1-p:.3f}

  ╔══════════════════════════════════════════╗
  ║  方向: {'▲ BUY ' if signal['direction']=='BUY' else ('▼ SELL' if signal['direction']=='SELL' else '─ HOLD')}                        ║
  ║  置信: {'■■■ 强' if conf>0.15 else ('■■ 中' if conf>0.08 else '■ 弱')} ({conf:.3f})                    ║
  ║                                         ║
  ║  止损: {signal['stop_loss']:.0f}    目标: {signal['target']:.0f}         ║
  ║  仓位: {signal['position']}                      ║
  ╚══════════════════════════════════════════╝

  融合方法: {signal['method']}
''')

    if signal.get('wf_p') is not None:
        print(f'  WF概率: {signal["wf_p"]:.3f}')
    if signal.get('wf_ext_p') is not None:
        print(f'  WF扩展概率: {signal["wf_ext_p"]:.3f}')
    if signal.get('full_p') is not None:
        print(f'  全量概率: {signal["full_p"]:.3f}')
    print(f'  动态仓位: {signal["position"]} (波动率调整后)')

    if ml_results:
        print(f'  ML验证:  fwd10 AUC={ml_results[10]["mean_auc"]:.2f}  Acc={ml_results[10]["mean_acc"]:.1%}')

    print(f'\n  【市场状态】')
    for label, detail in states:
        print(f'    · {label}: {detail}')

    print(f'\n  【关键特征】')
    for k, v in signal.get('top_features', [])[:5]:
        print(f'    {k:20s} {v:.3f}')

    if signal.get('new_feature_importance', 0) > 0.02:
        print(f'  【新增特征贡献】 {signal["new_feature_importance"]:.1%} 总重要性')
        for k, v in signal.get('new_features_detail', {}).items():
            print(f'    {k:20s} {v:.3f}')

    print(f'\n{"=" * 68}')
    print('⚠️ 统计模型输出,仅供参考,不构成投资建议.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--force', action='store_true', help='强制重建增强特征')
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    # 1. 构建增强特征
    feat = build_enhanced_features(force=args.force or args.rebuild)

    # 2. 集成预测
    signal_data = ensemble_predict(feat)

    # 3. 市场状态
    states = market_state(feat)

    # 4. ML验证
    from gold_tide_ml import load_features, walk_forward
    base_feat = load_features(FEAT_PATH)
    ml_results = {fwd: walk_forward(base_feat, fwd) for fwd in (5, 10, 20)}

    # 5. 行情数据
    from gold_tide_engine import load_data, compute_atr
    df = load_data(DATA_PATH)
    close = df['Close'].values.astype(float)
    dates = df['Date'].values.astype('datetime64[D]')
    atr_arr = compute_atr(df, 20)
    atr = atr_arr[-1]
    price = close[-1]

    # 6. 组装信号
    direction = signal_data['direction']
    if direction == 'BUY':
        sl = price - 2.5 * atr
        target = price + 3.0 * atr
    elif direction == 'SELL':
        sl = price + 2.5 * atr
        target = price - 3.0 * atr
    else:
        sl = price - 2 * atr
        target = price + 2 * atr

    conf = signal_data['confidence']
    pos = signal_data.get('position_pct', '15-20%')

    signal = {
        'date': str(dates[-1]),
        'price': price,
        **signal_data,
        'stop_loss': sl,
        'target': target,
        'position': pos,
    }

    # 7. 输出
    print_signal(signal, states, ml_results)

    # 如果是重建模式，也运行完整流水线
    if args.rebuild:
        import subprocess
        subprocess.run([sys.executable, 'run_pipeline.py', '--rebuild'])

    return signal, states, ml_results


if __name__ == '__main__':
    main()
