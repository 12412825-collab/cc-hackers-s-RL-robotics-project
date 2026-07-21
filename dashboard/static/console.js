const cfg = window.CONSOLE_CONFIG;
const state = {angle: 0, throttle: 0, mode: 'user', recording: false, telemetry: {}, lastTelemetry: 0};
const $ = id => document.getElementById(id);
const history = {speed: [], cte: [], angular: []};
const trail = [], keys = new Set(), MAX = 140;
let heading = 0, lastPosition = null;

let ws = null, reconnectTimer = null;
function connectWebSocket() {
  clearTimeout(reconnectTimer);
  $('socket-text').textContent = '正在连接';
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/wsDrive`);
  ws.onopen = () => {
    $('socket-dot').classList.add('online');
    $('socket-text').textContent = '实时连接';
    render();
  };
  ws.onclose = () => {
    $('socket-dot').classList.remove('online');
    $('socket-text').textContent = '连接断开，正在重连';
    state.lastTelemetry = 0;
    state.angle = 0;
    state.throttle = 0;
    moveStick(0, 0);
    render();
    reconnectTimer = setTimeout(connectWebSocket, 1000);
  };
  ws.onerror = () => { $('socket-text').textContent = '连接异常，正在重试'; };
  ws.onmessage = event => {
    let data;
    try { data = JSON.parse(event.data); } catch (_) { return; }
    if (data.driveMode !== undefined) state.mode = data.driveMode;
    if (data.recording !== undefined) state.recording = data.recording;
    if (data.telemetry) { state.telemetry = data.telemetry; state.lastTelemetry = Date.now(); }
    render(data.num_records);
  };
}
connectWebSocket();

function send(extra = {}) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({angle: state.angle, throttle: state.throttle, drive_mode: state.mode, recording: state.recording, ...extra})); }
function brake() { state.angle = 0; state.throttle = 0; state.recording = false; send(); moveStick(0, 0); render(); }
function mode(value) { state.mode = value; send(); render(); }
document.querySelectorAll('[data-mode]').forEach(button => button.onclick = () => mode(button.dataset.mode));
$('brake').onclick = brake;
$('record').onclick = () => { state.recording = !state.recording; send(); render(); };
function showSimulationMessage(message, error = false) {
  const element = $('simulation-message');
  if (!element) return;
  element.textContent = message;
  element.classList.toggle('error', error);
  element.hidden = false;
  clearTimeout(showSimulationMessage.timer);
  showSimulationMessage.timer = setTimeout(() => { element.hidden = true; }, 3500);
}
const refreshedMessage = sessionStorage.getItem('console-refresh-message');
if (refreshedMessage) {
  sessionStorage.removeItem('console-refresh-message');
  setTimeout(() => showSimulationMessage(refreshedMessage), 50);
}
$('reset-simulation').onclick = async () => {
  brake();
  $('reset-simulation').disabled = true;
  showSimulationMessage('正在重置车辆…');
  try {
    const response = await fetch('/api/simulation', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'reset'})
    });
    const responseText = await response.text();
    let result;
    try { result = JSON.parse(responseText); }
    catch (_) {
      throw new Error(response.status === 404
        ? '当前仍是旧后端，请重启 donkey_webots 控制器后再试'
        : `后端返回了非 JSON 响应（HTTP ${response.status}）`);
    }
    if (!response.ok || !result.ok) throw new Error(result.message || '重置失败');
    trail.length = 0;
    lastPosition = null;
    heading = 0;
    showSimulationMessage(result.confirmed
      ? '✓ Webots 已确认重置，位置应回到 X≈-3、Z≈0'
      : '重置请求已发送，但尚未收到 Webots 确认');
  } catch (error) {
    showSimulationMessage(`重置失败：${error.message}`, true);
  } finally {
    $('reset-simulation').disabled = false;
  }
};
$('refresh-console').onclick = () => {
  brake();
  sessionStorage.setItem('console-refresh-message', '✓ 控制台已刷新并重新连接');
  const url = new URL(location.href);
  url.searchParams.set('_refresh', Date.now().toString());
  location.replace(url.toString());
};

const joy = $('joystick'), stick = $('stick'); let dragging = false;
function moveStick(x, y) { stick.style.transform = `translate(${x * 90}px,${-y * 90}px)`; }
function point(event) { const r=joy.getBoundingClientRect(),p=event.touches?event.touches[0]:event; state.angle=Math.max(-1,Math.min(1,(p.clientX-r.left-r.width/2)/(r.width*.36))); state.throttle=Math.max(-1,Math.min(1,(r.top+r.height/2-p.clientY)/(r.height*.36))); moveStick(state.angle,state.throttle); send(); render(); }
joy.onpointerdown=e=>{dragging=true;joy.setPointerCapture(e.pointerId);point(e)};
joy.onpointermove=e=>{if(dragging)point(e)};
joy.onpointerup=()=>{dragging=false;brake()};

function driveKeys(){const f=keys.has('arrowup')||keys.has('w'),b=keys.has('arrowdown')||keys.has('s'),l=keys.has('arrowleft')||keys.has('a'),r=keys.has('arrowright')||keys.has('d');state.throttle=f?.55:b?-.4:0;state.angle=l?-.6:r?.6:0;if(f||b||l||r){moveStick(state.angle,state.throttle);send();render()}else brake()}
document.onkeydown=e=>{const k=e.key.toLowerCase();if(e.code==='Space'){e.preventDefault();keys.clear();brake();return}if(['arrowup','arrowdown','arrowleft','arrowright','w','a','s','d'].includes(k)){e.preventDefault();keys.add(k);driveKeys()}if(!e.repeat&&k==='r'){$('record').click()}if(!e.repeat&&k==='u')mode('user')};
document.onkeyup=e=>{if(keys.delete(e.key.toLowerCase()))driveKeys()};
window.addEventListener('blur',()=>{keys.clear();brake()});

document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
  const god=button.dataset.view==='god', camera=$('camera-view');
  $('god-view').hidden=!god;camera.hidden=god;
  $('view-title').textContent=god?'上帝视角':'车载视角';
  document.querySelectorAll('[data-view]').forEach(item=>item.classList.toggle('active',item===button));
  if(god){showSimulationMessage('✓ 已切换到上帝视角');return;}
  if(camera.complete&&camera.naturalWidth>0)showSimulationMessage(`✓ 车载视角已加载（${camera.naturalWidth}×${camera.naturalHeight}）`);
  else showSimulationMessage('正在加载车载视频…');
});
$('camera-view').addEventListener('load',()=>showSimulationMessage(`✓ 车载视频连接成功（${$('camera-view').naturalWidth}×${$('camera-view').naturalHeight}）`));
$('camera-view').addEventListener('error',()=>showSimulationMessage('车载视频加载失败，请检查 /video 摄像头流',true));

function drawGodView(){
  const c=$('god-view'),x=c.getContext('2d'),w=c.width,h=c.height,wx=v=>70+(v+4)/8*(w-140),wz=v=>h/2-v/.9*(h-100);
  const g=x.createLinearGradient(0,0,0,h);g.addColorStop(0,'#132942');g.addColorStop(1,'#07101c');x.fillStyle=g;x.fillRect(0,0,w,h);
  x.fillStyle='#292d35';x.fillRect(wx(-4),wz(.3),wx(4)-wx(-4),wz(-.3)-wz(.3));
  x.strokeStyle='#ffd05b';x.lineWidth=4;[.3,-.3].forEach(z=>{x.beginPath();x.moveTo(wx(-4),wz(z));x.lineTo(wx(4),wz(z));x.stroke()});
  x.setLineDash([18,14]);x.strokeStyle='#eef4fa';x.lineWidth=3;x.beginPath();x.moveTo(wx(-4),wz(0));x.lineTo(wx(4),wz(0));x.stroke();x.setLineDash([]);
  [{x:2,z:0,w:.10,d:.10},{x:-.3,z:.20,w:.16,d:.10},{x:.75,z:-.20,w:.16,d:.10}].forEach((o,i)=>{const px=wx(o.x),pz=wz(o.z),ow=o.w/8*(w-140),od=o.d/.9*(h-100);x.fillStyle='#ff654f';x.shadowColor='#ff654f';x.shadowBlur=14;x.fillRect(px-ow/2,pz-od/2,ow,od);x.shadowBlur=0;x.fillStyle='#ffd9d4';x.font='bold 13px Segoe UI';x.fillText(`障碍 ${i+1}`,px-24,pz-od/2-10)});
  if(trail.length>1){x.strokeStyle='#39d7e5aa';x.lineWidth=4;x.beginPath();trail.forEach((p,i)=>i?x.lineTo(wx(p.x),wz(p.z)):x.moveTo(wx(p.x),wz(p.z)));x.stroke()}
  const t=state.telemetry||{},px=wx(Number.isFinite(+t.x)?+t.x:-3),pz=wz(Number.isFinite(+t.z)?+t.z:0);x.save();x.translate(px,pz);x.rotate(-heading);x.fillStyle='#4d83ff';x.shadowColor='#39d7e5';x.shadowBlur=18;x.fillRect(-19,-13,38,26);x.shadowBlur=0;x.fillStyle='#ff5f6d';x.fillRect(11,-13,8,26);x.fillStyle='#fff';x.beginPath();x.moveTo(27,0);x.lineTo(15,-8);x.lineTo(15,8);x.fill();x.restore();
  x.fillStyle='#a9bdd7';x.font='14px Segoe UI';x.fillText('起点',wx(-3)-18,wz(-.38));x.fillText(`车辆 X ${n(t.x,2)} m · Z ${n(t.z,2)} m`,24,30);
}
function updatePose(t){const x=Number(t.x),z=Number(t.z);if(!Number.isFinite(x)||!Number.isFinite(z))return;if(lastPosition){const dx=x-lastPosition.x,dz=z-lastPosition.z;if(Math.hypot(dx,dz)>.002)heading=Math.atan2(dz,dx)}lastPosition={x,z};const p=trail.at(-1);if(!p||Math.hypot(x-p.x,z-p.z)>.015){trail.push({x,z});if(trail.length>240)trail.shift()}}
function n(value,digits=3){const number=Number(value);return (Number.isFinite(number)?number:0).toFixed(digits)}
function text(id,value){const element=$(id);if(element)element.textContent=value}
function render(records){
  const t=state.telemetry||{},v=state.throttle*cfg.maxLinear,w=state.angle*cfg.maxAngular,live=state.lastTelemetry>0;
  if(live)updatePose(t);drawGodView();
  text('v-command',`${n(v)} m/s`);text('w-command',`${n(w)} rad/s`);$('v-meter').value=v;$('w-meter').value=w;
  text('speed',n(t.speed));text('cte',n(t.cte));text('distance',Number(t.distance)>=0?n(t.distance,1):'—');text('residual',n(t.residual));text('left-speed',`${n(t.leftSpeed,2)} rad/s`);text('right-speed',`${n(t.rightSpeed,2)} rad/s`);text('position',`${n(t.x,2)} / ${n(t.z,2)} m`);text('acl',(t.acl||[0,0,0]).map(v=>n(v,2)).join(' / ')+' g');text('gyr',(t.gyr||[0,0,0]).map(v=>n(v,2)).join(' / ')+' °/s');
  if(records!==undefined){text('sample-count',`${records} 条记录`);text('dataset',`${records} 条记录`)}
  text('mode-badge',({user:'手动',local_angle:'自动转向',local:'全自动'})[state.mode]||state.mode);text('record-badge',state.recording?'● 正在录制':'未录制');$('record-badge').style.color=state.recording?'#ff7180':'';$('record').textContent=state.recording?'停止录制':'开始录制';document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
  const viewState=$('view-state');if(viewState)viewState.classList.toggle('hidden',live);text('sensor-health',live?'传感器在线':'等待数据');text('telemetry-age',live?'< 0.1 s':'—');
  const cte=Math.abs(Number(t.cte)||0),distance=Number(t.distance),cteCard=$('cte-card'),distanceCard=$('distance-card');if(cteCard)cteCard.classList.toggle('alert',cte>.25);if(distanceCard)distanceCard.classList.toggle('danger-alert',distance>=0&&distance<35);text('safety',distance>=0&&distance<35?'前方障碍':cte>.25?'偏离赛道':'正常');
  if(live&&t.speed!==undefined){history.speed.push(+t.speed||0);history.cte.push(+t.cte||0);history.angular.push(+t.angular||0);Object.values(history).forEach(a=>{if(a.length>MAX)a.shift()});drawChart()}
}
function drawChart(){const c=$('chart'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);x.strokeStyle='#263953';for(let i=1;i<4;i++){x.beginPath();x.moveTo(0,h*i/4);x.lineTo(w,h*i/4);x.stroke()}[['speed','#3cdb93',.3],['cte','#ffbd52',.5],['angular','#4d83ff',2]].forEach(([key,color,scale])=>{x.beginPath();x.strokeStyle=color;x.lineWidth=2;history[key].forEach((v,i)=>{const px=i/(MAX-1)*w,py=h/2-v/scale*h*.42;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()})}
render();setInterval(()=>{if(state.lastTelemetry&&Date.now()-state.lastTelemetry>5000){state.lastTelemetry=0;render()}},500);
