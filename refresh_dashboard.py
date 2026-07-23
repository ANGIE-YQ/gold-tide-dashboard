"""
refresh_dashboard.py —— 自动刷新看板
====================================
1. 从新浪拉取最新AU0日线,追加到CSV
2. 重建潮汐特征层(增量)
3. 重新训练/加载校准模型
4. 计算当前信号
5. 生成 docs/index.html 看板
6. 供 GitHub Actions 每日自动运行
====================================
"""
import os, sys, json, urllib.request, io
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'gold_AU0_daily.csv')
FEAT = os.path.join(BASE, 'gold_features.csv')
DOCS = os.path.join(BASE, 'docs')
os.makedirs(DOCS, exist_ok=True)

CST = timezone(timedelta(hours=8))

# ======================== 1. 数据刷新 ========================
def fetch_latest():
    """从新浪拉沪金AU0最新日线,增量追加到本地CSV"""
    print('[1/5] 拉取最新数据...')
    url = ('https://stock2.finance.sina.com.cn/futures/api/json.php/'
           'InnerFuturesNewService.getDailyKLine?symbol=AU0')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=30).read().decode('gbk', 'ignore')
    j = json.loads(raw)
    new = pd.DataFrame(j)
    new = new.rename(columns={'d':'Date','o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'})
    for c in ['Open','High','Low','Close','Volume']:
        new[c] = pd.to_numeric(new[c], errors='coerce')
    new['Date'] = pd.to_datetime(new['Date'])
    new = new.dropna(subset=['Close']).sort_values('Date')

    # 加载本地已有数据
    if os.path.exists(DATA):
        old = pd.read_csv(DATA, parse_dates=['Date'])
        last_date = pd.Timestamp(old['Date'].max())
        append = new[new['Date'] > last_date]
        if len(append) == 0:
            print(f'  无新数据,本地最新: {last_date.date()}')
            return old
        combined = pd.concat([old, append], ignore_index=True)
    else:
        combined = new

    combined = combined.drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
    combined.to_csv(DATA, index=False)
    print(f'  更新: {len(combined)}行, 最新: {combined.Date.max().date()}, 新增{len(append) if os.path.exists(DATA) else len(combined)}行')
    return combined

# ======================== 2. 重建特征层 ========================
def rebuild_features(df):
    """重建特征层(全量,因为潮汐分段依赖全局数据)"""
    print('[2/5] 重建特征层...')
    sys.path.insert(0, BASE)
    from gold_tide_engine import compute_atr, detect_tides
    from gold_tide_score import score_all_tides
    from gold_tide_features import build_features

    atr = compute_atr(df, 20)
    big, small, big_pv, small_pv = detect_tides(df, 3.0, 1.0, atr)
    score_all_tides(df, big, small, atr)
    feat = build_features(df, atr, big, small, small_pv, save_csv=True)
    print(f'  特征: {feat.shape[0]}行 x {feat.shape[1]}列')
    return feat, big, small, atr

# ======================== 3. 训练/加载模型 ========================
def ensure_model(feat, df):
    """确保校准模型存在且是最新的"""
    print('[3/5] 准备校准模型...')
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import pickle

    model_path = os.path.join(BASE, 'gold_tide_calibrated_model.pkl')
    skip = {'Date','Close','fwd5','fwd10','fwd20'}
    dead = {'ip_small_cnt','ip_small_up_ratio','ip_last_small_dir','ip_trend_slope'}
    fc = [c for c in feat.columns if c not in skip and c not in dead]

    y = (feat['fwd10'] > 0).astype(int).values
    ok = ~feat['fwd10'].isna().values
    X = feat[fc].values

    # 总是重新训练以确保使用最新数据
    base = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42)
    base.fit(X[ok], y[ok])
    cal = CalibratedClassifierCV(base, method='sigmoid', cv=5)
    cal.fit(X[ok], y[ok])

    with open(model_path, 'wb') as f:
        pickle.dump({'model': cal, 'features': fc, 'dead': list(dead)}, f)

    # 当前预测
    p_cur = float(cal.predict_proba(X[-1:].reshape(1,-1))[0,1])
    print(f'  模型就绪, 当前P(涨)={p_cur:.3f}')
    return cal, fc, p_cur, X[-1:]

# ======================== 4. 提取看板数据 ========================
def extract_dashboard_data(df, feat, big, small, atr_arr, p_cur):
    """提取所有看板需要的JSON数据"""
    print('[4/5] 提取看板数据...')
    close = df['Close'].values.astype(float)
    dates = df['Date'].astype(str).values
    n = len(close)

    # 进行中段小潮汐
    last_tide = big[-1]
    ip_start = last_tide['end_idx']
    ip_small = [s for s in small if s['start_idx'] >= ip_start]
    ip_data = []
    for s in ip_small:
        ip_data.append({
            'id': s['tide_id'], 'dir': s['direction'],
            'from': float(s['start_price']), 'to': float(s['end_price']),
            'height': float(s['height']), 'days': int(s['duration_bars']),
            'energy': float(s['energy']),
        })

    # 潮汐结构
    tide_list = []
    for leg in big[-10:]:
        tide_list.append({
            'id': leg['tide_id'], 'dir': leg['direction'],
            'start': str(leg['start_date'])[:10], 'end': str(leg['end_date'])[:10],
            'from': float(leg['start_price']), 'to': float(leg['end_price']),
            'height': float(leg['height']), 'days': int(leg['duration_bars']),
            'score': float(leg.get('momentum_score', 50)),
        })

    # 动能趋势
    mom_trend = []
    for leg in big[-10:]:
        mom_trend.append({
            'tide': leg['tide_id'], 'dir': leg['direction'],
            'score': float(leg.get('momentum_score', 50)),
        })

    # 最近60天价格
    recent60 = []
    for i in range(max(0, n-60), n):
        recent60.append({'date': str(dates[i])[:10], 'close': float(close[i])})

    # 市场状态
    last = feat.iloc[-1]
    atr = atr_arr[-1]
    price = close[-1]
    ma20 = float(np.mean(close[-20:]))

    # 方向
    if p_cur > 0.55:
        direction = 'BUY'
        stop_loss = price - 2.5 * atr
        target = price + 3.0 * atr
    elif p_cur < 0.45:
        direction = 'SELL'
        stop_loss = price + 2.5 * atr
        target = price - 3.0 * atr
    else:
        direction = 'HOLD'
        stop_loss = price - 2 * atr
        target = price + 2 * atr

    conf = abs(p_cur - 0.5)
    vol = float(last.get('vol_regime', 1.0))
    base_pos = 0.20 if conf > 0.15 else (0.12 if conf > 0.08 else 0.05)
    adj_pos = np.clip(base_pos / min(vol, 1.5), 0.03, 0.25)

    data = {
        'generated': datetime.now(CST).strftime('%Y-%m-%d %H:%M CST'),
        'current': {
            'date': str(pd.Timestamp(dates[-1]).date()),
            'price': float(price),
            'atr': float(atr),
            'ma20': float(ma20),
            'high20': float(close[-20:].max()),
            'low20': float(close[-20:].min()),
        },
        'signal': {
            'direction': direction,
            'p_up': round(p_cur, 4),
            'confidence': round(conf, 4),
            'stop': round(stop_loss, 0),
            'target': round(target, 0),
            'position': f'{adj_pos*100:.0f}%',
        },
        'market_state': {
            'rsi': float(last.get('rsi_14', 50)),
            'bull': bool(last.get('regime_bull', 0.5) > 0.5),
            'vol_regime': float(vol),
            'vs_ma50': float(last.get('price_vs_ma50', 0)) * 100,
            'ip_ret': float(last.get('ip_ret', 0)) * 100,
            'ip_energy_trend': float(last.get('ip_energy_trend', 0)),
        },
        'recent60': recent60,
        'tides': tide_list,
        'momentum_trend': mom_trend,
        'in_progress_small': ip_data,
    }
    return data

# ======================== 5. 生成HTML ========================
def generate_html(data):
    """生成自包含HTML看板"""
    print('[5/5] 生成 docs/index.html...')

    with open(os.path.join(BASE, 'docs', 'index.html'), 'w', encoding='utf-8') as f:
        f.write(generate_html_content(data))

    # 也保存data.js供参考
    with open(os.path.join(DOCS, 'data.js'), 'w', encoding='utf-8') as f:
        f.write('const DASHBOARD_DATA = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';')

    print(f'  看板已生成: docs/index.html')
    print(f'  数据文件: docs/data.js')

    # 也更新模拟器页面（用最新回测数据）
    try:
        import subprocess
        subprocess.run([sys.executable, 'trade_simulator.py', '--mode', 'backtest', '--capital', '100000'],
                       cwd=BASE, capture_output=True, timeout=180)
        subprocess.run([sys.executable, 'build_simulator_page.py'], cwd=BASE, capture_output=True, timeout=30)
        print(f'  模拟器已更新: docs/simulator.html')
    except Exception as e:
        print(f'  模拟器更新跳过: {e}')


def generate_html_content(D):
    """生成自包含HTML, 嵌入数据 + 实时价格拉取"""
    import json as _json
    data_json = _json.dumps(D, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Gold Tide · 黄金潮汐交易看板</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--t2:#8b949e;--t3:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d2991d;--up:#238636;--down:#da3633}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:16px;max-width:960px;margin:0 auto}}
h1{{font-size:20px}}h2{{font-size:15px;color:var(--t2);margin:16px 0 10px;border-bottom:1px solid var(--border);padding-bottom:4px}}
.live-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}}
.live-dot.on{{background:var(--green)}} .live-dot.off{{background:var(--yellow)}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}}
.big-num{{font-size:30px;font-weight:700}} .unit{{font-size:13px;color:var(--t2)}}
.label{{font-size:11px;color:var(--t2);margin-bottom:2px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600}}
.b-buy{{background:var(--up);color:#fff}} .b-sell{{background:var(--down);color:#fff}} .b-hold{{background:var(--yellow);color:#000}}
.row{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #1c2128;font-size:13px}}
.row:last-child{{border-bottom:none}} .row .k{{color:var(--t2)}} .row .v{{font-weight:600}}
.bar-w{{background:#1c2128;border-radius:3px;height:14px;overflow:hidden;width:100px;display:inline-block}}
.bar-f{{height:100%;border-radius:3px;background:var(--t3)}}
.p-bar{{height:6px;border-radius:3px;background:#1c2128;margin:10px 0;position:relative}}
.p-fill{{height:100%;border-radius:3px;position:absolute;left:0;background:linear-gradient(90deg,var(--red),var(--yellow),var(--green))}}
.p-mid{{position:absolute;top:-4px;width:2px;height:14px;background:var(--text);left:50%}}
.guide{{background:#1a2332;border-left:3px solid var(--t3);padding:10px 14px;margin:12px 0;font-size:12px;line-height:1.7;border-radius:0 4px 4px 0}}
.guide b{{color:var(--t3)}}
.footer{{text-align:center;color:var(--t2);font-size:11px;margin-top:20px}}
.price-update{{font-size:10px;color:var(--t2);margin-top:4px}}
.btn-refresh{{display:inline-block;padding:4px 12px;background:var(--card);border:1px solid var(--border);border-radius:5px;color:var(--t3);cursor:pointer;font-size:12px;margin-left:8px;transition:all .2s}}
.btn-refresh:hover{{border-color:var(--t3);background:#1a2332}}
.btn-refresh:disabled{{opacity:.5;cursor:wait}}
.toast{{position:fixed;top:20px;right:20px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 20px;font-size:13px;z-index:999;display:none;max-width:350px}}
.toast.show{{display:block}}
#pw-gate{{position:fixed;top:0;left:0;width:100%;height:100%;background:var(--bg);z-index:9999;display:flex;align-items:center;justify-content:center;flex-direction:column}}
#pw-gate input{{padding:10px 16px;font-size:16px;background:var(--card);border:1px solid var(--border);border-radius:6px;color:var(--text);width:240px;text-align:center;outline:none}}
#pw-gate input:focus{{border-color:var(--t3)}}
#pw-gate .hint{{color:var(--t2);font-size:12px;margin-top:8px}}
#setup-guide{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:9998;align-items:center;justify-content:center}}
#setup-guide.show{{display:flex}}
#setup-guide .box{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:24px;max-width:440px;text-align:center}}
#setup-guide .box a{{color:var(--t3)}}
</style>
</head>
<body>

<div id="pw-gate">
  <h2 style="margin-bottom:16px">Gold Tide</h2>
  <input type="password" id="pw-input" placeholder="请输入访问密码" onkeydown="if(event.key==='Enter')checkPw()">
  <div class="hint" id="pw-hint"></div>
</div>

<div id="main-content" style="display:none">

<div style="display:flex;justify-content:space-between;align-items:center">
<div>
<h1>Gold Tide · 黄金潮汐交易看板</h1>
<div style="font-size:12px;color:var(--t2)">沪金AU0 | <span class="live-dot on" id="live-dot"></span><span id="live-status">实时</span> | 模型更新: {D['generated']}</div>
<button class="btn-refresh" onclick="triggerRefresh()" id="btn-refresh">🔄 更新模型数据</button>
</div>
<div style="text-align:right">
<div style="font-size:28px;font-weight:700" id="live-price">--</div>
<div class="price-update" id="live-change">加载中...</div>
</div>
</div>

<div id="toast" class="toast"></div>

<div id="setup-guide" class="show" style="display:none">
  <div class="box">
    <h3 style="margin-bottom:12px">⚡ 一键更新设置（仅需一次）</h3>
    
    <p style="font-size:13px;color:var(--t2);margin:8px 0"><b>方法一：Fine-grained Token（推荐）</b></p>
    <p style="font-size:12px;color:var(--t2)">1. 打开 <a href="https://github.com/settings/tokens?type=beta" target="_blank">GitHub Token页面</a></p>
    <p style="font-size:12px;color:var(--t2)">2. 点 <b>Generate new token</b> → 选 <b>ANGIE-YQ/gold-tide-dashboard</b></p>
    <p style="font-size:12px;color:var(--t2)">3. Repository permissions → <b>Actions: Read and write</b></p>
    <p style="font-size:12px;color:var(--t2)">4. Contents: <b>Read and write</b></p>
    <p style="font-size:12px;color:var(--t2)">5. 点 Generate, 复制Token(以 github_pat_ 开头)</p>

    <p style="font-size:13px;color:var(--t2);margin:12px 0"><b>方法二：Classic Token</b></p>
    <p style="font-size:12px;color:var(--t2)">打开 <a href="https://github.com/settings/tokens/new?scopes=workflow,repo&description=Gold+Tide" target="_blank">这个链接</a> → 直接点底部 Generate → 复制Token(以 ghp_ 开头)</p>

    <input type="text" id="token-input" placeholder="粘贴Token到这里" style="width:100%;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);margin:8px 0;font-size:13px">
    <button onclick="saveToken()" style="padding:8px 24px;background:var(--t3);color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:14px">保存并启用</button>
    <p style="font-size:11px;color:var(--t2);margin-top:8px;cursor:pointer" onclick="document.getElementById('setup-guide').style.display='none'">以后再说</p>
  </div>
</div>

<div class="grid3" style="margin-top:12px">
<div class="card"><div class="label">模型价格</div><div class="big-num">{D['current']['price']:.0f}<span class="unit">元/克</span></div></div>
<div class="card"><div class="label">20日均线</div><div class="big-num">{D['current']['ma20']:.0f}<span class="unit">vs {((D['current']['price']/D['current']['ma20']-1)*100):+.1f}%</span></div></div>
<div class="card"><div class="label">ATR波动率</div><div class="big-num">{D['current']['atr']:.1f}<span class="unit">20日</span></div></div>
</div>

<h2>核心信号</h2>
<div class="card" style="text-align:center;padding:20px">
<div class="badge {'b-buy' if D['signal']['direction']=='BUY' else ('b-sell' if D['signal']['direction']=='SELL' else 'b-hold')}" style="font-size:16px;padding:4px 16px">{D['signal']['direction']}</div>
<div style="font-size:42px;font-weight:800;margin:10px 0;color:{'var(--green)' if D['signal']['direction']=='BUY' else ('var(--red)' if D['signal']['direction']=='SELL' else 'var(--yellow)')}">P(涨)={D['signal']['p_up']:.3f}</div>
<div style="color:var(--t2)">置信度: {D['signal']['confidence']*100:.1f}% {'(强)' if D['signal']['confidence']>0.15 else ('(中)' if D['signal']['confidence']>0.08 else '(弱)')}</div>
<div class="p-bar" style="max-width:280px;margin:10px auto"><div class="p-fill" style="width:{D['signal']['p_up']*100}%"></div><div class="p-mid"></div></div>
<div style="display:flex;gap:24px;justify-content:center;margin-top:12px">
<div><span class="label">止损</span><br><strong style="color:var(--red)">{D['signal']['stop']:.0f}</strong></div>
<div><span class="label">目标</span><br><strong style="color:var(--green)">{D['signal']['target']:.0f}</strong></div>
<div><span class="label">仓位</span><br><strong>{D['signal']['position']}</strong></div>
</div>
</div>

<h2>市场状态</h2>
<div class="grid4" id="market-state"></div>

<h2>潮汐动能趋势(近10大潮汐)</h2>
<div class="card"><svg id="mom-svg" width="100%" height="160"></svg></div>

<h2>进行中段微观结构(能量标注)</h2>
<div class="card"><svg id="ip-svg" width="100%" height="180"></svg></div>

<h2>价格走势(近60日, 含潮汐标注)</h2>
<div class="card"><svg id="price-svg" width="100%" height="260"></svg></div>

<h2>阅读指南</h2>
<div class="guide">
<b>核心信号</b> — P(涨)>0.55做多, <0.45做空。置信度>0.15强信号。右上角实时价格来自公开API(15秒刷新)。<br>
<b>潮汐图</b> — 蓝=上涨潮汐, 红=下跌潮汐, 标注为双名法。R6a=R大潮汐第a小潮汐。<br>
<b>进行中段</b> — 最后已确认潮汐结束后的"未完成"段。E值=能量(高度×时间), 比较同向潮汐能量判断趋势衰竭。<br>
<b>模型预测未来10个交易日方向, 非日内, 非长期。</b>
</div>

<div class="footer">AI统计模型输出,仅供参考,不构成投资建议 | Data: NeoData/Sina · Model: XGBoost+Platt · Private</div>

</div><!-- /main-content -->

<script>
// ======================== PASSWORD GATE ========================
const PW_HASH = '572d50108697d44e04e11daa498fe94096a92be49bacea20d07dfee2892f1cad';

async function sha256(m) {{
  const buf = new TextEncoder().encode(m);
  const hash = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hash)).map(b=>b.toString(16).padStart(2,'0')).join('');
}}

async function checkPw() {{
  const input = document.getElementById('pw-input').value;
  const h = await sha256(input);
  if (h === PW_HASH) {{
    document.getElementById('pw-gate').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';
    localStorage.setItem('gt_auth', '1');
    initCharts();
  }} else {{
    document.getElementById('pw-hint').textContent = '密码错误';
    document.getElementById('pw-input').value = '';
  }}
}}

function initCharts() {{
  renderMarketState(); renderPriceChart(); renderMomChart(); renderIPChart();
  fetchLivePrice(); setInterval(fetchLivePrice, 15000);
}}

// Auto-login if previously authenticated
if (localStorage.getItem('gt_auth') === '1') {{
  document.getElementById('pw-gate').style.display = 'none';
  document.getElementById('main-content').style.display = 'block';
}}

// ======================== EMBEDDED MODEL DATA ========================
const M = {data_json};

// ======================== LIVE PRICE FETCH (15s interval) ========================
async function fetchLivePrice() {{
  try {{
    // Use Sina Finance API for AU0 futures
    const resp = await fetch('https://hq.sinajs.cn/list=au0', {{
      headers: {{ 'Referer': 'https://finance.sina.com.cn' }}
    }});
    const text = await resp.text();
    // Parse: var hq_str_au0="...";
    const match = text.match(/"([^"]+)"/);
    if (!match) throw new Error('parse failed');
    const parts = match[1].split(',');
    // parts[0]=name, parts[3]=current price, parts[2]=change%, parts[1]=open
    const price = parseFloat(parts[3]);
    const changePct = parseFloat(parts[2]);
    if (isNaN(price)) throw new Error('price NaN');
    document.getElementById('live-price').textContent = price.toFixed(2);
    const sign = changePct >= 0 ? '+' : '';
    document.getElementById('live-change').innerHTML = 
      `<span style="color:${{changePct>=0?'var(--green)':'var(--red)'}}">${{sign}}${{changePct.toFixed(2)}}%</span>`;
    document.getElementById('live-dot').className = 'live-dot on';
    document.getElementById('live-status').textContent = '实时';
  }} catch(e) {{
    document.getElementById('live-dot').className = 'live-dot off';
    document.getElementById('live-status').textContent = '离线';
    console.log('Live price fetch failed:', e.message);
  }}
}}

// ======================== RENDER STATIC CHARTS ========================
function renderMarketState() {{
  const s = M.market_state;
  const items = [
    ['牛/熊', (s.bull?'牛市':'熊市/调整'), s.bull?'var(--green)':'var(--red)', 'vs 250日均线'],
    ['波动率', s.vol_regime.toFixed(1)+'x常态', s.vol_regime<0.7?'var(--green)':(s.vol_regime>1.5?'var(--red)':'var(--text)'), s.vol_regime<0.7?'低波动(可加仓)':(s.vol_regime>1.5?'高波动(降仓)':'正常')],
    ['RSI(14)', s.rsi.toFixed(0), s.rsi<30?'var(--green)':(s.rsi>70?'var(--red)':'var(--text)'), s.rsi<30?'超卖':(s.rsi>70?'超买':'中性')],
    ['vs 50日均', s.vs_ma50.toFixed(1)+'%', s.vs_ma50<0?'var(--red)':'var(--green)', s.vs_ma50<-5?'深度超卖':''],
  ];
  document.getElementById('market-state').innerHTML = items.map(([l,v,c,d])=>`<div class="card"><div class="label">${{l}}</div><strong style="color:${{c}}">${{v}}</strong><div style="font-size:10px;color:var(--t2)">${{d}}</div></div>`).join('');
}}

function renderPriceChart() {{
  const svg=document.getElementById('price-svg'), w=svg.clientWidth||900, h=260, pad={{t:15,r:30,b:35,l:50}}, pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const data=M.recent60, prices=data.map(d=>d.close), pmin=Math.min(...prices)*0.98, pmax=Math.max(...prices)*1.02;
  const sY=v=>pad.t+ph*(1-(v-pmin)/(pmax-pmin)), sX=i=>pad.l+(i/(data.length-1))*pw;
  let html='';
  for(let i=0;i<5;i++){{const y=pad.t+ph*i/4,v=pmax-(pmax-pmin)*i/4;html+=`<line x1="${{pad.l}}" x2="${{pad.l+pw}}" y1="${{y}}" y2="${{y}}" stroke="#1c2128"/><text x="${{pad.l-6}}" y="${{y+4}}" text-anchor="end" fill="#8b949e" font-size="10">${{v.toFixed(0)}}</text>`}}
  let dM=''; data.forEach((d,i)=>{{const x=sX(i),y=sY(d.close);dM+=(i===0?'M':'L')+`${{x}},${{y}} `}}); html+=`<path d="${{dM}}" fill="none" stroke="#c9d1d9" stroke-width="1.5"/>`;
  // MA20 overlay
  let maD=''; for(let i=19;i<data.length;i++){{const avg=data.slice(i-19,i+1).reduce((s,d)=>s+d.close,0)/20;maD+=(i===19?'M':'L')+`${{sX(i)}},${{sY(avg)}} `}} html+=`<path d="${{maD}}" fill="none" stroke="#d2991d" stroke-width="1" stroke-dasharray="3,2"/>`;
  // Last price marker
  const lx=sX(data.length-1),ly=sY(data[data.length-1].close);
  html+=`<circle cx="${{lx}}" cy="${{ly}}" r="3" fill="#58a6ff"/><text x="${{lx}}" y="${{ly-8}}" text-anchor="middle" fill="#58a6ff" font-size="10" font-weight="700">${{data[data.length-1].close.toFixed(0)}}</text>`;
  svg.innerHTML=html;
}}

function renderMomChart() {{
  const svg=document.getElementById('mom-svg'), w=svg.clientWidth||900, h=160, pad={{t:15,r:20,b:35,l:35}}, pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const data=M.momentum_trend.slice(-10);
  let html=`<line x1="${{pad.l}}" x2="${{pad.l+pw}}" y1="${{pad.t+ph/2}}" y2="${{pad.t+ph/2}}" stroke="#30363d" stroke-dasharray="3,2"/><text x="${{pad.l-5}}" y="${{pad.t+ph/2+4}}" text-anchor="end" fill="#8b949e" font-size="9">50</text>`;
  data.forEach((d,i)=>{{
    const bw=pw/(data.length*2.5), x=pad.l+(i/(data.length-1))*pw, y=pad.t+ph*(1-d.score/100);
    const col=d.dir==='up'?'#238636':'#da3633';
    html+=`<rect x="${{x-bw/2}}" y="${{y}}" width="${{bw}}" height="${{pad.t+ph-y}}" fill="${{col}}" opacity="0.8" rx="2"/>`;
    html+=`<text x="${{x}}" y="${{y-5}}" text-anchor="middle" fill="${{col}}" font-size="9">${{d.score.toFixed(0)}}</text>`;
    html+=`<text x="${{x}}" y="${{pad.t+ph+15}}" text-anchor="middle" fill="#8b949e" font-size="9">${{d.tide}}</text>`;
  }});
  svg.innerHTML=html;
}}

function renderIPChart() {{
  const svg=document.getElementById('ip-svg'), w=svg.clientWidth||900, h=180, pad={{t:20,r:30,b:35,l:50}}, pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
  const ts=M.in_progress_small;
  if(!ts.length){{svg.innerHTML='<text x="50%" y="50%" text-anchor="middle" fill="#8b949e">暂无数据</text>';return}}
  let cum=0; const pos=ts.map(t=>{{const s=cum;cum+=t.days;return{{start:s,end:s+t.days,...t}}}});
  const allP=ts.flatMap(t=>[t.from,t.to]), pmin=Math.min(...allP)*0.96, pmax=Math.max(...allP)*1.02;
  const sY=v=>pad.t+ph*(1-(v-pmin)/(pmax-pmin)), sX=v=>pad.l+(v/cum)*pw;
  let html='';
  for(let i=0;i<5;i++){{const y=pad.t+ph*i/4;html+=`<line x1="${{pad.l}}" x2="${{pad.l+pw}}" y1="${{y}}" y2="${{y}}" stroke="#1c2128"/>`}}
  pos.forEach(t=>{{
    const x1=sX(t.start),x2=sX(t.end),y1=sY(t.from),y2=sY(t.to),col=t.dir==='up'?'#238636':'#da3633';
    html+=`<line x1="${{x1}}" y1="${{y1}}" x2="${{x2}}" y2="${{y2}}" stroke="${{col}}" stroke-width="3"/>`;
    html+=`<text x="${{(x1+x2)/2}}" y="${{Math.min(y1,y2)-7}}" text-anchor="middle" fill="${{col}}" font-size="10">${{t.id}}</text>`;
    html+=`<text x="${{(x1+x2)/2}}" y="${{Math.max(y1,y2)+13}}" text-anchor="middle" fill="#8b949e" font-size="9">E=${{t.energy.toFixed(0)}}</text>`;
  }});
  // Energy trend annotation
  const upT=pos.filter(t=>t.dir==='up');
  if(upT.length>=2){{
    const e1=upT[upT.length-2].energy,e2=upT[upT.length-1].energy;
    const tx=sX(pos[pos.length-1].end)+10;
    html+=`<text x="${{tx}}" y="${{pad.t+20}}" fill="${{e2<e1?'#3fb950':'#f85149'}}" font-size="10">${{e2<e1?'能量渐弱→可能反转':'能量增强→趋势延续'}}</text>`;
    html+=`<text x="${{tx}}" y="${{pad.t+36}}" fill="#8b949e" font-size="9">${{upT[upT.length-2].energy.toFixed(0)}}→${{e2.toFixed(0)}}(${{((e2/e1-1)*100).toFixed(0)}}%)</text>`;
  }}
  svg.innerHTML=html;
}}

// ======================== REFRESH BUTTON ========================
function showToast(msg, isOk) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderColor = isOk ? 'var(--green)' : 'var(--yellow)';
  t.className = 'toast show';
  setTimeout(() => t.className = 'toast', 4000);
}}

async function triggerRefresh() {{
  const btn = document.getElementById('btn-refresh');
  let token = localStorage.getItem('gh_token');
  
  if (!token) {{
    // No token - show setup guide
    document.getElementById('setup-guide').style.display = 'block';
    return;
  }}
  
  btn.disabled = true;
  btn.textContent = '触发中...';
  
  try {{
    const resp = await fetch('https://api.github.com/repos/ANGIE-YQ/gold-tide-dashboard/actions/workflows/update-dashboard.yml/dispatches', {{
      method:'POST',
      headers: {{'Authorization':'token '+token, 'Accept':'application/vnd.github+json'}},
      body: JSON.stringify({{ref:'main'}})
    }});
    if (resp.status === 204) {{
      showToast('✅ 已触发! 约1-2分钟后刷新页面。', true);
      btn.textContent = '刷新页面';
      btn.onclick = () => location.reload();
    }} else {{
      const err = await resp.text();
      if (resp.status === 403) {{
        showToast('403权限不足。请重新生成Token,确保勾选Actions和Contents权限。', false);
        document.getElementById('setup-guide').style.display = 'flex';
      }} else if (resp.status === 404) {{
        showToast('工作流未配置,请先将WORKFLOW-SETUP.yml移到.github/workflows/', false);
      }} else {{
        showToast('触发失败('+resp.status+'), Token可能过期。双击标题重新设置。', false);
      }}
      btn.disabled = false;
      btn.textContent = '更新模型数据';
    }}
  }} catch(e) {{
    showToast('网络错误: '+e.message, false);
    btn.disabled = false;
    btn.textContent = '更新模型数据';
  }}
}}

function saveToken() {{
  const t = document.getElementById('token-input').value.trim();
  if (t.startsWith('ghp_') || t.startsWith('github_pat_')) {{
    localStorage.setItem('gh_token', t);
    document.getElementById('setup-guide').style.display = 'none';
    showToast('已保存! 试试点更新按钮。', true);
  }} else {{
    showToast('格式不对。Fine-grained以github_pat_开头, Classic以ghp_开头', false);
  }}
}}

// Init
window.onload=function(){{
  if (localStorage.getItem('gt_auth') === '1') {{
    initCharts();
  }}
}};
window.onresize=function(){{
  renderPriceChart();
  renderMomChart();
  renderIPChart();
}};
</script>
</body>
</html>'''


# ======================== MAIN ========================
def main():
    start = datetime.now()
    print(f'=== 黄金潮汐看板刷新 ===')
    print(f'时间: {start.strftime("%Y-%m-%d %H:%M:%S")}')

    df = fetch_latest()
    feat, big, small, atr_arr = rebuild_features(df)
    model, fc, p_cur, _ = ensure_model(feat, df)
    data = extract_dashboard_data(df, feat, big, small, atr_arr, p_cur)
    generate_html(data)

    elapsed = (datetime.now() - start).total_seconds()
    print(f'\n刷新完成, 耗时 {elapsed:.0f}s')
    print(f'信号: {data["signal"]["direction"]} P={data["signal"]["p_up"]:.3f}')
    return data


if __name__ == '__main__':
    main()
