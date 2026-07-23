"""将交易模拟器嵌入主看板，作为可展开面板"""
import json, re

# 读取回测数据
with open('sim_result.json', 'r', encoding='utf-8') as f:
    sim_data = json.load(f)

eq = sim_data['equity_curve']
step = max(1, len(eq) // 300)
eq_sampled = eq[::step]
if eq_sampled[-1] != eq[-1]:
    eq_sampled.append(eq[-1])

trades_20 = sim_data['trades'][-20:]
perf = sim_data['perf']

# 读取现有index.html
with open('docs/index.html', 'r', encoding='utf-8') as f:
    index = f.read()

# 1. 在header按钮组中添加模拟器按钮
old_btn = '<button class="btn-refresh" id="btn-refresh" onclick="triggerRefresh()">🔄 更新模型数据</button>'
new_btn = old_btn + '\n      <button class="btn-refresh" onclick="toggleSimulator()" style="color:var(--t5);border-color:var(--t5)">📊 交易模拟</button>'
index = index.replace(old_btn, new_btn)

# 2. 在 footer 前插入模拟器面板
sim_panel = f'''
<div id="simulator-panel" style="display:none">
  <div class="card" style="margin-top:16px;max-width:100%;overflow:hidden">
    <h3 style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      📊 交易模拟器 (Walk-Forward回测 2014-2026)
      <span onclick="toggleSimulator()" style="cursor:pointer;color:var(--t1);font-size:20px">&times;</span>
    </h3>
    <p style="font-size:12px;color:var(--t5);margin-bottom:12px;display:flex;flex-wrap:wrap;gap:16px">
      <span>总收益 <b style="color:var(--t3)">+{perf['total_return_pct']:.1f}%</b></span>
      <span>年化 <b style="color:var(--t2)">+{perf['cagr_pct']:.1f}%</b></span>
      <span>夏普 <b style="color:var(--text)">{perf['sharpe']:.2f}</b></span>
      <span>最大回撤 <b style="color:var(--t4)">{perf['max_drawdown_pct']:.1f}%</b></span>
      <span>胜率 <b style="color:var(--t5)">{perf['win_rate_pct']:.0f}%</b></span>
      <span>盈亏比 <b>{perf['profit_factor']:.1f}</b></span>
      <span>{perf['n_trades']}笔</span>
    </p>
    <div style="height:300px;margin-bottom:12px"><canvas id="equity-canvas"></canvas></div>
    <div style="overflow-x:auto;max-height:280px;overflow-y:auto">
      <table class="trade-table" style="font-size:12px;min-width:500px">
        <thead><tr>
          <th>入场</th><th>出场</th><th>方向</th><th>盈亏%</th><th>盈亏额</th><th>原因</th><th>信号P</th>
        </tr></thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
    <p style="font-size:11px;color:var(--t1);margin-top:10px;line-height:1.6">
      <b>交易规则</b>: 模型P(涨)>0.60做多 / P<0.40做空 | Kelly仓位公式(上限25%) | 止损2.5×ATR / 止盈3.0×ATR | 信号反转时平仓<br>
      <b>验证方式</b>: Walk-Forward扩展窗口(1500训/200测), 无前视偏差 | 校准模型(XGBoost+Platt) | 覆盖{perf['n_days']}个交易日<br>
      <span style="color:var(--t4)">⚠ 模拟回测仅用于验证模型信号质量, 不构成交易建议。实际交易需考虑滑点、手续费、流动性等。</span>
    </p>
  </div>
</div>
'''

footer_pos = index.find('<div class="footer">')
if footer_pos > 0:
    index = index[:footer_pos] + sim_panel + '\n' + index[footer_pos:]

# 3. 在 </body> 前加入模拟器数据和脚本
sim_script = f'''
<script>
const SIM_EQ = {json.dumps(eq_sampled, ensure_ascii=False)};
const SIM_TRADES = {json.dumps(trades_20, ensure_ascii=False)};

function toggleSimulator() {{
  const panel = document.getElementById('simulator-panel');
  if (!panel) return;
  if (panel.style.display === 'none' || !panel.style.display) {{
    panel.style.display = 'block';
    setTimeout(() => {{ drawSimEquity(); renderSimTrades(); }}, 100);
  }} else {{
    panel.style.display = 'none';
  }}
}}

function drawSimEquity() {{
  const c = document.getElementById('equity-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.parentElement.clientWidth;
  const H = 300;
  c.width = W * 2; c.height = H * 2;
  c.style.width = W + 'px'; c.style.height = H + 'px';
  ctx.scale(2, 2);
  
  if (!SIM_EQ.length) return;
  const pad = {{l: 50, r: 20, t: 15, b: 30}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const vals = SIM_EQ.map(function(d) {{ return d.equity; }});
  const mn = Math.min.apply(null, vals) * 0.97;
  const mx = Math.max.apply(null, vals) * 1.03;
  
  function sx(i) {{ return pad.l + (i / (SIM_EQ.length - 1)) * pw; }}
  function sy(v) {{ return pad.t + (1 - (v - mn) / (mx - mn)) * ph; }}
  
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (var i = 0; i <= 4; i++) {{
    var v = mn + (mx - mn) * i / 4;
    var y = sy(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
    ctx.fillText((v / 1000).toFixed(0) + 'k', 5, y + 4);
  }}
  
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.beginPath();
  SIM_EQ.forEach(function(d, i) {{
    var x = sx(i), y = sy(d.equity);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
  
  ctx.lineTo(sx(SIM_EQ.length - 1), sy(mn));
  ctx.lineTo(sx(0), sy(mn));
  ctx.fillStyle = 'rgba(63,185,80,0.05)'; ctx.fill();
  
  SIM_EQ.forEach(function(d, i) {{
    if (d.in_position) {{
      ctx.fillStyle = d.position_dir === 'long' ? 'rgba(63,185,80,0.08)' : 'rgba(248,81,73,0.08)';
      ctx.fillRect(sx(i) - 2, pad.t, 3, ph);
    }}
  }});
  
  ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
  ctx.fillText(SIM_EQ[0].date, pad.l, H - 8);
  var lastDate = SIM_EQ[SIM_EQ.length - 1].date;
  ctx.fillText(lastDate, W - pad.r - ctx.measureText(lastDate).width - 5, H - 8);
}}

function renderSimTrades() {{
  var tb = document.getElementById('trade-body');
  if (!tb) return;
  var h = '';
  var labels = {{take_profit: '止盈', stop_loss: '止损', signal_flip: '信号反转', close_all: '平仓'}};
  var colors = {{take_profit: 'badge-tp', stop_loss: 'badge-sl', signal_flip: 'badge-sf'}};
  var trades = SIM_TRADES.slice().reverse();
  for (var i = 0; i < trades.length; i++) {{
    var t = trades[i];
    var cls = t.pnl > 0 ? 'win' : 'loss';
    var dc = t.dir === 'long' ? 'green' : 'red';
    var pc = t.pnl > 0 ? 'green' : 'red';
    h += '<tr class="' + cls + '"><td>' + t.entry + '</td><td>' + t.exit + '</td><td class="' + dc + '">' + (t.dir === 'long' ? '做多' : '做空') + '</td><td class="' + pc + '">' + t.pnl_pct.toFixed(2) + '%</td><td class="' + pc + '">' + t.pnl.toFixed(0) + '</td><td><span class="badge ' + (colors[t.reason] || '') + '">' + (labels[t.reason] || t.reason) + '</span></td><td>' + t.signal_p.toFixed(3) + '</td></tr>';
  }}
  tb.innerHTML = h;
}}
</script>
'''

index = index.replace('</body>', sim_script + '\n</body>')

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(index)

print(f'docs/index.html: {len(index):,} chars')
print(f'Equity points: {len(eq_sampled)}')
print(f'Trades shown: {len(trades_20)}')
print('DONE - simulator embedded in main dashboard')
