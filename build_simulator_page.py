#!/usr/bin/env python3
"""生成交易模拟器HTML看板 + 整合到主看板"""

import json, base64
from datetime import datetime

# 读取回测数据
with open('sim_result.json', 'r', encoding='utf-8') as f:
    sim_data = json.load(f)

# 采样权益曲线(最多300点)
eq = sim_data['equity_curve']
step = max(1, len(eq) // 300)
eq_sampled = eq[::step]
if eq_sampled[-1] != eq[-1]:
    eq_sampled.append(eq[-1])

trades = sim_data['trades']
perf = sim_data['perf']

# 最近20笔交易
recent_trades = trades[-20:]

# 年度收益
from collections import defaultdict
yearly = defaultdict(float)
for t in trades:
    yr = t['entry'][:4]
    yearly[yr] += t['pnl_pct']

now = datetime.now().strftime('%Y-%m-%d %H:%M')

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>交易模拟器 · 黄金潮汐</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--t1:#8b949e;--t2:#58a6ff;--t3:#3fb950;--t4:#f85149;--t5:#d2991d}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:20px}}
.nav{{display:flex;gap:12px;margin-bottom:20px;align-items:center}}
.nav a,.nav span{{color:var(--t1);text-decoration:none;font-size:14px}}
.nav a:hover{{color:var(--t2)}}
.nav h1{{font-size:22px;color:var(--t3);margin-right:auto}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px}}
.card h3{{font-size:14px;color:var(--t1);margin-bottom:12px;font-weight:400}}
.big{{font-size:28px;font-weight:700}}
.green{{color:var(--t3)}}
.red{{color:var(--t4)}}
.yellow{{color:var(--t5)}}
.blue{{color:var(--t2)}}
.label{{font-size:12px;color:var(--t1);margin-top:4px}}
.chart-area{{width:100%;height:380px;position:relative}}
.trade-table{{width:100%;border-collapse:collapse;font-size:13px}}
.trade-table th{{text-align:left;color:var(--t1);font-weight:400;padding:6px 10px;border-bottom:1px solid var(--border);font-size:12px}}
.trade-table td{{padding:6px 10px;border-bottom:1px solid var(--border)}}
.trade-table tr:last-child td{{border:none}}
.win{{background:rgba(63,185,80,0.08)}}
.loss{{background:rgba(248,81,73,0.08)}}
.btn{{display:inline-block;padding:8px 20px;background:var(--t3);color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:14px;transition:all .2s}}
.btn:hover{{opacity:.85}}
.btn-secondary{{background:var(--border);color:var(--text)}}
.control-row{{display:flex;gap:12px;align-items:center;margin:16px 0;flex-wrap:wrap}}
.control-row label{{font-size:13px;color:var(--t1)}}
.control-row input{{padding:6px 10px;background:var(--bg);border:1px solid var(--border);border-radius:5px;color:var(--text);width:80px;font-size:13px}}
.control-row input:focus{{border-color:var(--t2);outline:none}}
.tab-bar{{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid var(--border)}}
.tab{{padding:8px 16px;cursor:pointer;color:var(--t1);font-size:14px;border-bottom:2px solid transparent;transition:all .2s}}
.tab:hover{{color:var(--text)}}
.tab.active{{color:var(--t2);border-bottom-color:var(--t2)}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px}}
.badge-tp{{background:rgba(63,185,80,0.15);color:var(--t3)}}
.badge-sl{{background:rgba(248,81,73,0.15);color:var(--t4)}}
.badge-sf{{background:rgba(88,166,255,0.15);color:var(--t2)}}
.footer{{text-align:center;color:var(--t1);font-size:12px;margin-top:30px;padding-top:20px;border-top:1px solid var(--border)}}
#live-price{{font-size:24px;font-weight:700}}
.price-update{{font-size:11px;color:var(--t1)}}
</style>
</head>
<body>

<div class="nav">
  <h1>📊 交易模拟器</h1>
  <a href="index.html">← 返回看板</a>
</div>

<!-- 绩效总览 -->
<div class="grid4">
  <div class="card">
    <h3>总收益率</h3>
    <div class="big green">{perf['total_return_pct']:+.1f}%</div>
    <div class="label">初始 {perf['initial_capital']:,.0f} → 终值 {perf['final_capital']:,.0f}</div>
  </div>
  <div class="card">
    <h3>年化收益</h3>
    <div class="big blue">{perf['cagr_pct']:+.1f}%</div>
    <div class="label">夏普比率 {perf['sharpe']}</div>
  </div>
  <div class="card">
    <h3>最大回撤</h3>
    <div class="big red">{perf['max_drawdown_pct']:.1f}%</div>
    <div class="label">盈亏比 {perf['profit_factor']}</div>
  </div>
  <div class="card">
    <h3>胜率</h3>
    <div class="big yellow">{perf['win_rate_pct']:.0f}%</div>
    <div class="label">{perf['n_trades']}笔交易 · 持仓{perf['position_ratio_pct']:.0f}%</div>
  </div>
</div>

<!-- 标签切换 -->
<div class="tab-bar">
  <div class="tab active" onclick="switchTab('equity')">权益曲线</div>
  <div class="tab" onclick="switchTab('trades')">交易记录</div>
  <div class="tab" onclick="switchTab('live')">实时模拟</div>
  <div class="tab" onclick="switchTab('params')">参数调整</div>
</div>

<!-- 权益曲线 -->
<div id="tab-equity" class="tab-content active">
  <div class="card">
    <h3>权益曲线 (回测: 2014-2026)</h3>
    <div class="chart-area"><canvas id="equity-canvas"></canvas></div>
  </div>
  
  <div class="grid2" style="margin-top:16px">
    <div class="card">
      <h3>年度收益分布</h3>
      <div class="chart-area" style="height:250px"><canvas id="yearly-canvas"></canvas></div>
    </div>
    <div class="card">
      <h3>退出原因分布</h3>
      <div class="chart-area" style="height:250px"><canvas id="reason-canvas"></canvas></div>
    </div>
  </div>
</div>

<!-- 交易记录 -->
<div id="tab-trades" class="tab-content">
  <div class="card">
    <h3>最近20笔交易</h3>
    <div style="overflow-x:auto">
    <table class="trade-table">
      <thead><tr>
        <th>入场</th><th>出场</th><th>方向</th><th>入场价</th><th>出场价</th>
        <th>盈亏%</th><th>盈亏额</th><th>原因</th><th>信号P</th>
      </tr></thead>
      <tbody>
'''
for t in reversed(recent_trades):
    cls = 'win' if t['pnl'] > 0 else 'loss'
    dir_color = 'green' if t['dir'] == 'long' else 'red'
    reason_badge = {'take_profit': 'badge-tp', 'stop_loss': 'badge-sl', 'signal_flip': 'badge-sf', 'close_all': ''}.get(t['reason'], '')
    reason_label = {'take_profit': '止盈', 'stop_loss': '止损', 'signal_flip': '信号反转', 'close_all': '强制平仓'}.get(t['reason'], t['reason'])
    html += f'''        <tr class="{cls}">
          <td>{t['entry']}</td><td>{t['exit']}</td>
          <td class="{dir_color}">{'做多' if t['dir']=='long' else '做空'}</td>
          <td>{t['entry_price']:.1f}</td><td>{t['exit_price']:.1f}</td>
          <td class="{'green' if t['pnl']>0 else 'red'}">{t['pnl_pct']:+.2f}%</td>
          <td class="{'green' if t['pnl']>0 else 'red'}">{t['pnl']:+,.0f}</td>
          <td><span class="badge {reason_badge}">{reason_label}</span></td>
          <td>{t['signal_p']:.3f}</td>
        </tr>\n'''

html += '''      </tbody>
    </table>
    </div>
  </div>
</div>

<!-- 实时模拟 -->
<div id="tab-live" class="tab-content">
  <div class="card">
    <h3>实时模拟信号</h3>
    <div style="display:flex;gap:40px;align-items:center;flex-wrap:wrap">
      <div>
        <div style="font-size:13px;color:var(--t1);margin-bottom:6px">当前价格</div>
        <div id="live-price">--</div>
        <div class="price-update" id="live-change">加载中...</div>
      </div>
      <div>
        <div style="font-size:13px;color:var(--t1);margin-bottom:6px">模型P(涨)</div>
        <div id="live-p" class="big" style="font-size:22px">--</div>
      </div>
      <div>
        <div style="font-size:13px;color:var(--t1);margin-bottom:6px">信号方向</div>
        <div id="live-dir" class="big" style="font-size:22px">--</div>
      </div>
    </div>
    <div id="live-suggestion" style="margin-top:16px;padding:12px;background:var(--bg);border-radius:8px;display:none"></div>
  </div>
  
  <div class="card" style="margin-top:16px">
    <h3>交易规则说明</h3>
    <div style="font-size:13px;color:var(--t1);line-height:1.8">
      <b style="color:var(--t3)">开仓</b>: 模型P(涨) > 0.60 → 做多; P < 0.40 → 做空<br>
      <b style="color:var(--t4)">止损</b>: 2.5 × ATR (平均真实波幅)<br>
      <b style="color:var(--t3)">止盈</b>: 3.0 × ATR<br>
      <b style="color:var(--t2)">仓位</b>: Kelly公式 × 置信度 × 0.5, 上限25%<br>
      <b style="color:var(--t5)">退出</b>: 止损/止盈触发, 或信号方向反转时平仓<br>
      <br>
      <b>重要</b>: 实时模拟仅用于测试模型信号质量, 不构成交易建议。实际交易需考虑滑点、手续费、流动性等。
    </div>
  </div>
</div>

<!-- 参数调整 -->
<div id="tab-params" class="tab-content">
  <div class="card">
    <h3>回测参数调整</h3>
    <p style="font-size:13px;color:var(--t1);margin-bottom:16px">调整参数后点击"重新回测"在本地运行。以下为当前回测参数:</p>
    
    <div class="grid2">
      <div>
        <h4 style="font-size:13px;color:var(--t1);margin-bottom:8px">信号阈值</h4>
        <div class="control-row">
          <label>买入阈值 P ></label>
          <input type="number" id="buy-thr" value="0.60" step="0.01" min="0.5" max="0.95">
          <label>卖出阈值 P <</label>
          <input type="number" id="sell-thr" value="0.40" step="0.01" min="0.05" max="0.5">
        </div>
        
        <h4 style="font-size:13px;color:var(--t1);margin:16px 0 8px">风险控制</h4>
        <div class="control-row">
          <label>止损 ATR×</label>
          <input type="number" id="stop-atr" value="2.5" step="0.1" min="1" max="5">
          <label>止盈 ATR×</label>
          <input type="number" id="tp-atr" value="3.0" step="0.1" min="1" max="8">
        </div>
        
        <h4 style="font-size:13px;color:var(--t1);margin:16px 0 8px">仓位管理</h4>
        <div class="control-row">
          <label>初始资金</label>
          <input type="number" id="init-cap" value="100000" step="10000">
          <label>最大仓位</label>
          <input type="number" id="max-pos" value="25" step="5">%
        </div>
      </div>
      <div>
        <h4 style="font-size:13px;color:var(--t1);margin-bottom:8px">当前回测结果</h4>
        <div style="font-size:13px;color:var(--t1);line-height:2">
          阈值: P>{sim_data['params']['buy_threshold']} / P<{sim_data['params']['sell_threshold']}<br>
          止损: {sim_data['params']['stop_atr']}×ATR<br>
          止盈: {sim_data['params']['tp_atr']}×ATR<br>
          仓位上限: {sim_data['params']['max_position']*100:.0f}%<br>
          初始资金: {sim_data['params']['initial_capital']:,}<br>
          <br>
          <span style="color:var(--t5)">⚠ 参数调整后需在本地运行回测:<br>
          <code>python trade_simulator.py --mode backtest --capital 100000</code></span>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="footer">
  黄金潮汐交易模拟器 | 数据来源: NeoData/Sina | 模型: XGBoost + Platt校准 | Walk-Forward无前视偏差<br>
  <span style="color:var(--t4)">⚠ 模拟结果仅用于模型验证, 不构成交易建议。实际交易有风险。</span>
</div>

<script>
// EMBEDDED DATA
const EQ_DATA = {json.dumps(eq_sampled, ensure_ascii=False)};
const YEARLY = {json.dumps(dict(yearly), ensure_ascii=False)};
const REASONS = {json.dumps(perf['reason_stats'], ensure_ascii=False)};
const PERF = {json.dumps(perf, ensure_ascii=False)};

// ======================== Tab切换 ========================
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('{name}')"]`).classList.add('active');
  document.getElementById('tab-{name}').classList.add('active');
  if (name === 'equity') {{ drawEquity(); drawYearly(); drawReason(); }}
}}

// ======================== 权益曲线 ========================
function drawEquity() {{
  const canvas = document.getElementById('equity-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 380;
  canvas.width = W * 2; canvas.height = H * 2;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  ctx.scale(2, 2);
  
  if (!EQ_DATA.length) return;
  
  const pad = {{left:50,right:30,top:20,bottom:40}};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;
  
  const vals = EQ_DATA.map(d => d.equity);
  const minY = Math.min(...vals) * 0.98;
  const maxY = Math.max(...vals) * 1.02;
  
  const sx = i => pad.left + (i / (EQ_DATA.length - 1)) * pw;
  const sy = v => pad.top + (1 - (v - minY) / (maxY - minY)) * ph;
  
  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let v = minY; v <= maxY; v += (maxY-minY)/5) {{
    const y = sy(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W-pad.right, y); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui';
    ctx.fillText((v/1000).toFixed(0)+'k', 5, y+4);
  }}
  
  // Equity line
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.beginPath();
  EQ_DATA.forEach((d, i) => {{
    const x = sx(i), y = sy(d.equity);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
  
  // Fill
  ctx.lineTo(sx(EQ_DATA.length-1), sy(minY));
  ctx.lineTo(sx(0), sy(minY));
  ctx.fillStyle = 'rgba(63,185,80,0.05)';
  ctx.fill();
  
  // Labels
  const first = EQ_DATA[0], last = EQ_DATA[EQ_DATA.length-1];
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui';
  ctx.fillText(first.date, pad.left, H-10);
  ctx.fillText(last.date, W-pad.right-ctx.measureText(last.date).width, H-10);
  
  // Position markers
  EQ_DATA.forEach((d, i) => {{
    if (d.in_position) {{
      ctx.fillStyle = d.position_dir === 'long' ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)';
      ctx.fillRect(sx(i)-1, pad.top, 2, ph);
    }}
  }});
}}

// ======================== 年度收益 ========================
function drawYearly() {{
  const canvas = document.getElementById('yearly-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 250;
  canvas.width = W * 2; canvas.height = H * 2;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  ctx.scale(2, 2);
  
  const pad = {{left:45,right:20,top:20,bottom:35}};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;
  
  const yrs = Object.keys(YEARLY).sort();
  const vals = yrs.map(y => YEARLY[y] * 100); // to percentage
  const barW = Math.min(pw / yrs.length * 0.7, 40);
  const gap = pw / yrs.length;
  
  let lo = Math.min(0, ...vals) * 1.2;
  let hi = Math.max(0, ...vals) * 1.2;
  const sy = v => pad.top + (1 - (v - lo) / (hi - lo)) * ph;
  
  // Zero line
  const zy = sy(0);
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.left, zy); ctx.lineTo(W-pad.right, zy); ctx.stroke();
  
  yrs.forEach((yr, i) => {{
    const v = vals[i];
    const x = pad.left + i * gap + (gap - barW) / 2;
    const y = sy(Math.max(0, v));
    const h = Math.abs(sy(v) - sy(0));
    ctx.fillStyle = v >= 0 ? '#3fb950' : '#f85149';
    ctx.fillRect(x, y, barW, Math.max(h, 1));
    
    ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(yr, x + barW/2, H - 10);
    ctx.fillText(v.toFixed(1)+'%', x + barW/2, y - 5);
  }});
  ctx.textAlign = 'start';
  
  ctx.fillStyle = 'var(--t1)'; ctx.font = '11px system-ui';
  ctx.fillText('年度收益%', 5, 15);
}}

// ======================== 退出原因 ========================
function drawReason() {{
  const canvas = document.getElementById('reason-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth;
  const H = 250;
  canvas.width = W * 2; canvas.height = H * 2;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  ctx.scale(2, 2);
  
  const pad = {{left:80,right:30,top:20,bottom:20}};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;
  
  const labels = {{'take_profit':'止盈','stop_loss':'止损','signal_flip':'信号反转','close_all':'强制平仓'}};
  const colors = {{'take_profit':'#3fb950','stop_loss':'#f85149','signal_flip':'#58a6ff','close_all':'#8b949e'}};
  
  const items = Object.entries(REASONS).filter(([k]) => k !== 'close_all');
  const barH = Math.min(ph / items.length * 0.6, 30);
  const gap = ph / items.length;
  
  items.forEach(([k, s], i) => {{
    const y = pad.top + i * gap + (gap - barH) / 2;
    const w = Math.max(s.count / 420 * pw, 10);
    ctx.fillStyle = colors[k] || '#8b949e';
    ctx.beginPath(); ctx.roundRect(pad.left, y, w, barH, 3); ctx.fill();
    ctx.fillStyle = '#8b949e'; ctx.font = '12px system-ui';
    ctx.fillText(labels[k]||k, pad.left-70, y + barH/2 + 4);
    ctx.fillStyle = '#c9d1d9';
    ctx.fillText(s.count+'笔 (胜率'+s.win_rate.toFixed(0)+'% 均值'+s.avg_pnl_pct.toFixed(2)+'%)', pad.left + w + 8, y + barH/2 + 4);
  }});
}}

// ======================== 实时模拟 ========================
async function fetchLivePrice() {{
  try {{
    const resp = await fetch('https://hq.sinajs.cn/list=au0', {{headers:{{Referer:'https://finance.sina.com.cn'}}}});
    const text = await resp.text();
    const parts = text.split(',');
    if (parts.length > 3) {{
      const price = parseFloat(parts[3]);
      const prev = parseFloat(parts[2]);
      const change = ((price - prev) / prev * 100).toFixed(2);
      const dir = change >= 0 ? '+' : '';
      document.getElementById('live-price').textContent = price.toFixed(2);
      document.getElementById('live-change').textContent = dir+change+'% · ' + new Date().toLocaleTimeString();
      document.getElementById('live-price').className = change >= 0 ? 'green' : 'red';
      
      // Calculate suggestion based on current price
      updateSuggestion(price);
    }}
  }} catch(e) {{}}
}}

function updateSuggestion(price) {{
  // 使用嵌入的模型状态给出建议
  const lastSignal = EQ_DATA[EQ_DATA.length - 1];
  const dir = lastSignal.signal_p > 0.60 ? 'BUY' : (lastSignal.signal_p < 0.40 ? 'SELL' : 'HOLD');
  const p = lastSignal.signal_p;
  const atr = lastSignal.atr;
  
  document.getElementById('live-p').textContent = p.toFixed(4);
  document.getElementById('live-p').className = p > 0.5 ? 'big green' : 'big red';
  document.getElementById('live-p').style.fontSize = '22px';
  
  document.getElementById('live-dir').textContent = dir;
  document.getElementById('live-dir').className = dir === 'BUY' ? 'big green' : (dir === 'SELL' ? 'big red' : 'big yellow');
  document.getElementById('live-dir').style.fontSize = '22px';
  
  const sug = document.getElementById('live-suggestion');
  if (dir === 'HOLD') {{
    sug.style.display = 'block';
    sug.innerHTML = '<span style="color:var(--t5)">当前无明确信号, 建议观望</span>';
  }} else {{
    const isLong = dir === 'BUY';
    const sl = isLong ? price - 2.5 * atr : price + 2.5 * atr;
    const tp = isLong ? price + 3.0 * atr : price - 3.0 * atr;
    const conf = Math.abs(p - 0.5);
    const kelly = Math.max(0, (p > 0.5 ? (p*1.5-(1-p))/1.5 : ((1-p)*1.5-p)/1.5)) * 0.5;
    const posPct = Math.min(kelly * (conf/0.15), 0.25) * 100;
    
    sug.style.display = 'block';
    sug.innerHTML = `
      <div style="color:var(--text);font-weight:700;margin-bottom:8px">建议: ${dir === 'BUY' ? '做多' : '做空'}</div>
      <div style="display:flex;gap:20px;font-size:13px;color:var(--t1)">
        <div>入场: <b style="color:var(--text)">${price.toFixed(2)}</b></div>
        <div>止损: <b style="color:var(--t4)">${sl.toFixed(2)}</b></div>
        <div>止盈: <b style="color:var(--t3)">${tp.toFixed(2)}</b></div>
        <div>仓位: <b style="color:var(--t2)">${posPct.toFixed(1)}%</b></div>
      </div>
    `;
  }}
}}

// ======================== 初始化 ========================
window.onload = function() {{
  drawEquity();
  drawYearly();
  drawReason();
  fetchLivePrice();
  setInterval(fetchLivePrice, 15000);
}};
</script>
</body>
</html>'''

with open('docs/simulator.html', 'w', encoding='utf-8') as f:
    f.write(html)

# 在主看板添加模拟器链接
with open('docs/index.html', 'r', encoding='utf-8') as f:
    dashboard = f.read()

# 在导航栏添加链接
if 'simulator.html' not in dashboard:
    dashboard = dashboard.replace(
        '<h1 style="margin:0;color:var(--t3)">Gold Tide',
        '<h1 style="margin:0;color:var(--t3)"><a href="simulator.html" style="color:var(--t3);text-decoration:none">Gold Tide</a>'
    )
    # 在标题下方加入模拟器入口
    dashboard = dashboard.replace(
        '<div style="font-size:12px;color:var(--t2)">沪金AU0',
        '<div style="font-size:12px;color:var(--t2)">沪金AU0 | <a href="simulator.html" style="color:var(--t5)">📊 交易模拟器</a></div>\n<div style="font-size:12px;color:var(--t2)">'
    )

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(dashboard)

print(f'Simulator HTML: {len(html):,} chars')
print(f'Dashboard updated with simulator link')
print(f'Files: docs/simulator.html + docs/index.html')
