"""将实盘交易模拟器嵌入主看板，作为可展开面板"""
import json, re

# 读取实盘模拟数据（trade_simulator.py --mode live 的输出）
data_file = 'sim_live_result.json'
if not __import__('os').path.exists(data_file):
    data_file = 'sim_result.json'  # fallback

with open(data_file, 'r', encoding='utf-8') as f:
    live_data = json.load(f)

# 权益日志
eq_log = live_data.get('equity_log', [])
# 取最近300点
if len(eq_log) > 300:
    step = len(eq_log) // 300
    eq_sampled = eq_log[::step] + [eq_log[-1]]
else:
    eq_sampled = eq_log

# 最近交易
trades_20 = live_data.get('recent_trades', [])
perf = live_data.get('perf', {})
signal = live_data.get('signal', {})
position = live_data.get('position', None)

# 读取现有index.html
with open('docs/index.html', 'r', encoding='utf-8') as f:
    index = f.read()

# 1. 在header按钮组中添加模拟器按钮
old_btn = '<button class="btn-refresh" id="btn-refresh" onclick="triggerRefresh()">🔄 更新模型数据</button>'
new_btn = old_btn + '\n      <button class="btn-refresh" onclick="toggleSimulator()" style="color:var(--t5);border-color:var(--t5)">📊 交易模拟</button>'
index = index.replace(old_btn, new_btn)

# 2. 在 footer 前插入模拟器面板
# 构建持仓状态HTML
pos_html = ''
if position:
    p_type = '做多' if position['direction'] == 'long' else '做空'
    p_color = 'var(--t3)' if position['unrealized_pnl'] >= 0 else 'var(--t4)'
    pos_html = f'''
    <div style="margin:8px 0;padding:10px;background:var(--bg);border-radius:8px;border-left:3px solid {p_color}">
      <b style="font-size:14px">当前持仓: {p_type}</b>
      <span style="margin-left:16px;">入场价: <b>{position["entry_price"]}</b> (自{position["entry_date"]})</span>
      <span style="margin-left:16px;">浮动盈亏: <b style="color:{p_color}">{position["unrealized_pnl_pct"]:+.2f}%</b></span>
      <span style="margin-left:16px;">止损: <b style="color:var(--t4)">{position["stop_loss"]}</b></span>
      <span style="margin-left:16px;">止盈: <b style="color:var(--t3)">{position["take_profit"]}</b></span>
    </div>'''
elif signal.get('suggestion'):
    sug = signal['suggestion']
    pos_html = f'''
    <div style="margin:8px 0;padding:10px;background:var(--bg);border-radius:8px;border-left:3px solid var(--t2)">
      <b style="font-size:14px">今日建议: {sug["action"]}</b>
      <span style="margin-left:16px;">仓位: <b>{sug["position_pct"]}%</b></span>
      <span style="margin-left:16px;">止损: <b style="color:var(--t4)">{sug["stop_loss"]}</b></span>
      <span style="margin-left:16px;">止盈: <b style="color:var(--t3)">{sug["take_profit"]}</b></span>
    </div>'''

# 今日信号
sig_dir = signal.get('direction', 'HOLD')
sig_color = 'var(--t3)' if sig_dir == 'BUY' else ('var(--t4)' if sig_dir == 'SELL' else 'var(--t5)')

sim_panel = f'''
<div id="simulator-panel" style="display:none">
  <div class="card" style="margin-top:16px;max-width:100%;overflow:hidden">
    <h3 style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      📊 实盘交易模拟 <span style="font-size:11px;color:var(--t1);font-weight:400">(起始{perf.get("start_date","?")}，每日自动更新)</span>
      <span onclick="toggleSimulator()" style="cursor:pointer;color:var(--t1);font-size:20px">&times;</span>
    </h3>
    
    {pos_html}
    
    <div style="display:flex;gap:40px;align-items:center;margin:12px 0;flex-wrap:wrap">
      <div><span style="color:var(--t1);font-size:12px">今日信号</span><br><b style="font-size:22px;color:{sig_color}">{sig_dir}</b></div>
      <div><span style="color:var(--t1);font-size:12px">P(涨)</span><br><b style="font-size:16px">{signal.get('p_up',0):.4f}</b></div>
      <div><span style="color:var(--t1);font-size:12px">置信度</span><br><b style="font-size:16px">{signal.get('confidence',0):.4f}</b></div>
      <div><span style="color:var(--t1);font-size:12px">当前价</span><br><b style="font-size:16px">{signal.get('price',0):.2f}</b></div>
    </div>
    
    <p style="font-size:12px;color:var(--t5);margin-bottom:8px;display:flex;flex-wrap:wrap;gap:16px">
      <span>资金: <b>{perf.get('initial_capital',100000):,.0f}</b> → <b style="color:{'var(--t3)' if perf.get('total_return_pct',0)>=0 else 'var(--t4)'}">{perf.get('capital',100000):,.0f}</b></span>
      <span>总收益: <b style="color:{'var(--t3)' if perf.get('total_return_pct',0)>=0 else 'var(--t4)'}">{perf.get('total_return_pct',0):+.1f}%</b></span>
      <span>胜率: <b>{perf.get('win_rate_pct',0):.0f}%</b></span>
      <span>最大回撤: <b style="color:var(--t4)">{perf.get('max_drawdown_pct',0):.1f}%</b></span>
      <span>交易: <b>{perf.get('n_trades',0)}笔</b></span>
      <span>跟踪: <b>{perf.get('n_days',0)}天</b></span>
    </p>
    
    <div style="height:250px;margin-bottom:12px"><canvas id="equity-canvas"></canvas></div>
    
    <div style="overflow-x:auto;max-height:240px;overflow-y:auto">
      <table class="trade-table" style="font-size:12px;min-width:500px">
        <thead><tr>
          <th>入场</th><th>出场</th><th>方向</th><th>盈亏%</th><th>盈亏额</th><th>原因</th>
        </tr></thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
    
    <p style="font-size:11px;color:var(--t1);margin-top:10px;line-height:1.6">
      <b>规则</b>: P(涨)>0.60→做多 / P<0.40→做空 | Kelly仓位(上限25%) | 止损2.5×ATR / 止盈3.0×ATR | 信号反转平仓<br>
      <b>数据</b>: 校准模型(XGBoost+Platt) | 每日自动推进 | 无前视偏差 | 更新于 {live_data.get('last_update','?')}<br>
      <span style="color:var(--t4)">⚠ 模拟仅用于测试模型信号质量, 不构成交易建议。</span>
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
