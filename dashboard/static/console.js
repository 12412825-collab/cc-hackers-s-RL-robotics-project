const cfg = window.CONSOLE_CONFIG;
const state = {angle: 0, throttle: 0, mode: 'user', recording: false, brake: true, telemetry: {}};
const $ = id => document.getElementById(id);
const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/wsDrive`);
const history = {speed: [], cte: [], angular: []}, trail = [], keys = new Set(), MAX = 140;
let heading = 0, lastPosition = null;

ws.onopen = () => {$('socket-dot').classList.add('online'); $('socket-text').textContent = '实时连接'};
ws.onclose = () => {$('socket-dot').classList.remove('online'); $('socket-text').textContent = '连接断开'; brake()};
ws.onmessage = event => {
  const data = JSON.parse(event.data);
  if (data.driveMode !== undefined) state.mode = data.driveMode;
  if (data.recording !== undefined) state.recording = data.recording;
  if (data.telemetry) state.telemetry = data.telemetry;
  render(data.num_records);
};
function send(extra = {}) {if (ws.readyState === 1) ws.send(JSON.stringify({angle: state.angle, throttle: state.throttle, drive_mode: state.mode, recording: state.recording, ...extra}))}
function brake() {state.angle = 0; state.throttle = 0; state.brake = true; state.recording = false; send(); moveStick(0, 0); render()}
function mode(value) {state.mode = value; send(); render()}
document.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => mode(button.dataset.mode));
$('brake').onclick = brake;
$('record').onclick = () => {state.recording = !state.recording; send(); render()};

const joy = $('joystick'), stick = $('stick'); let dragging = false;
function moveStick(x, y) {stick.style.transform = `translate(${x * 90}px,${-y * 90}px)`}
function point(event) {const rect = joy.getBoundingClientRect(), pointer = event.touches ? event.touches[0] : event; const x = Math.max(-1, Math.min(1, (pointer.clientX - rect.left - rect.width / 2) / (rect.width * .36))); const y = Math.max(-1, Math.min(1, (rect.top + rect.height / 2 - pointer.clientY) / (rect.height * .36))); state.angle = x; state.throttle = y; state.brake = false; moveStick(x, y); send(); render()}
joy.onpointerdown = event => {dragging = true; joy.setPointerCapture(event.pointerId); point(event)};
joy.onpointermove = event => {if (dragging) point(event)};
joy.onpointerup = () => {dragging = false; brake()};

function driveKeys() {
  const forward = keys.has('arrowup') || keys.has('w'), back = keys.has('arrowdown') || keys.has('s');
  const left = keys.has('arrowleft') || keys.has('a'), right = keys.has('arrowright') || keys.has('d');
  state.throttle = forward ? .55 : back ? -.4 : 0;
  state.angle = left ? -.6 : right ? .6 : 0;
  if (forward || back || left || right) {state.brake = false; moveStick(state.angle, state.throttle); send(); render()} else brake();
}
document.onkeydown = event => {
  const key = event.key.toLowerCase();
  if (event.code === 'Space') {event.preventDefault(); keys.clear(); brake(); return}
  if (['arrowup','arrowdown','arrowleft','arrowright','w','a','s','d'].includes(key)) {event.preventDefault(); keys.add(key); driveKeys(); return}
  if (!event.repeat && key === 'r') {state.recording = !state.recording; send(); render()}
  if (!event.repeat && key === 'u') mode('user');
};
document.onkeyup = event => {if (keys.delete(event.key.toLowerCase())) driveKeys()};
window.addEventListener('blur', () => {keys.clear(); brake()});

document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => {
  const god = button.dataset.view === 'god';
  $('god-view').hidden = !god; $('camera-view').hidden = god;
  $('view-title').textContent = god ? '上帝视角' : '车载视角';
  document.querySelectorAll('[data-view]').forEach(item => item.classList.toggle('active', item === button));
});

function drawGodView() {
  const canvas = $('god-view'), ctx = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  const worldX = x => 70 + (x + 4) / 8 * (width - 140), worldZ = z => height / 2 - z / .9 * (height - 100);
  const gradient = ctx.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, '#132942'); gradient.addColorStop(1, '#07101c'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#292d35'; ctx.fillRect(worldX(-4), worldZ(.3), worldX(4)-worldX(-4), worldZ(-.3)-worldZ(.3));
  ctx.strokeStyle = '#ffd05b'; ctx.lineWidth = 4; [0.3,-0.3].forEach(z => {ctx.beginPath(); ctx.moveTo(worldX(-4),worldZ(z)); ctx.lineTo(worldX(4),worldZ(z)); ctx.stroke()});
  ctx.setLineDash([18,14]); ctx.strokeStyle = '#eef4fa'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(worldX(-4),worldZ(0)); ctx.lineTo(worldX(4),worldZ(0)); ctx.stroke(); ctx.setLineDash([]);
  const obstacles = [{x:2,z:0,w:.10,d:.10},{x:-.3,z:.20,w:.16,d:.10},{x:.75,z:-.20,w:.16,d:.10}];
  obstacles.forEach((o,index) => {const x=worldX(o.x),z=worldZ(o.z),w=o.w/8*(width-140),d=o.d/.9*(height-100); ctx.fillStyle='#ff654f'; ctx.shadowColor='#ff654f'; ctx.shadowBlur=16; ctx.fillRect(x-w/2,z-d/2,w,d); ctx.shadowBlur=0; ctx.fillStyle='#ffd9d4'; ctx.font='bold 13px Segoe UI'; ctx.fillText(`障碍 ${index+1}`,x-24,z-d/2-10)});
  if (trail.length > 1) {ctx.strokeStyle='#39d7e5aa'; ctx.lineWidth=4; ctx.beginPath(); trail.forEach((p,i) => i ? ctx.lineTo(worldX(p.x),worldZ(p.z)) : ctx.moveTo(worldX(p.x),worldZ(p.z))); ctx.stroke()}
  const t=state.telemetry||{}, x=worldX(Number(t.x)||-3), z=worldZ(Number(t.z)||0); ctx.save(); ctx.translate(x,z); ctx.rotate(-heading); ctx.fillStyle='#4d83ff'; ctx.shadowColor='#39d7e5'; ctx.shadowBlur=18; ctx.fillRect(-17,-12,34,24); ctx.shadowBlur=0; ctx.fillStyle='#ff5f6d'; ctx.fillRect(10,-12,7,24); ctx.fillStyle='#fff'; ctx.beginPath(); ctx.moveTo(24,0); ctx.lineTo(14,-7); ctx.lineTo(14,7); ctx.fill(); ctx.restore();
  ctx.fillStyle='#a9bdd7'; ctx.font='14px Segoe UI'; ctx.fillText('起点',worldX(-3)-18,worldZ(-.38)); ctx.fillText(`车辆 X ${Number(t.x||0).toFixed(2)} m · Z ${Number(t.z||0).toFixed(2)} m`,24,30);
}
function updatePose(t) {const x=Number(t.x),z=Number(t.z); if (!Number.isFinite(x)||!Number.isFinite(z)) return; if (lastPosition) {const dx=x-lastPosition.x,dz=z-lastPosition.z; if (Math.hypot(dx,dz)>.002) heading=Math.atan2(dz,dx)} lastPosition={x,z}; const previous=trail[trail.length-1]; if (!previous||Math.hypot(x-previous.x,z-previous.z)>.015) {trail.push({x,z}); if(trail.length>240)trail.shift()}}
function n(value, digits=3) {return Number(value || 0).toFixed(digits)}
function text(id, value) {$(id).textContent = value}
function render(records) {const t=state.telemetry||{},v=state.throttle*cfg.maxLinear,w=state.angle*cfg.maxAngular; updatePose(t); drawGodView(); text('v-command',`${n(v)} m/s`); text('w-command',`${n(w)} rad/s`); $('v-meter').value=v; $('w-meter').value=w; text('speed',n(t.speed)); text('cte',n(t.cte)); text('distance',t.distance>=0?n(t.distance,1):'—'); text('residual',n(t.residual)); text('left-speed',`${n(t.leftSpeed,2)} rad/s`); text('right-speed',`${n(t.rightSpeed,2)} rad/s`); text('position',`${n(t.x,2)} / ${n(t.z,2)} m`); text('acl',(t.acl||[0,0,0]).map(x=>n(x,2)).join(' / ')+' g'); text('gyr',(t.gyr||[0,0,0]).map(x=>n(x,2)).join(' / ')+' °/s'); if(records!==undefined){text('sample-count',`${records} samples`);text('dataset',`${records} 条记录`)} text('mode-badge',({user:'手动',local_angle:'自动转向',local:'全自动'})[state.mode]||state.mode); text('record-badge',state.recording?'● 正在录制':'未录制'); $('record-badge').style.color=state.recording?'#ff7180':''; $('record').textContent=state.recording?'停止录制':'开始录制'; document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode)); if(t.speed!==undefined){history.speed.push(+t.speed||0);history.cte.push(+t.cte||0);history.angular.push(+t.angular||0);Object.values(history).forEach(a=>{if(a.length>MAX)a.shift()});drawChart()}}
function drawChart(){const c=$('chart'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);x.strokeStyle='#263953';x.lineWidth=1;for(let i=1;i<4;i++){x.beginPath();x.moveTo(0,h*i/4);x.lineTo(w,h*i/4);x.stroke()}[['speed','#3cdb93',.3],['cte','#ffbd52',.5],['angular','#4d83ff',2]].forEach(([key,color,scale])=>{const a=history[key];x.beginPath();x.strokeStyle=color;x.lineWidth=2;a.forEach((v,i)=>{const px=i/(MAX-1)*w,py=h/2-v/scale*h*.42;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()})}
render();
