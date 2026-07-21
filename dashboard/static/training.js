(() => {
  const title = document.getElementById('job-title');
  const message = document.getElementById('job-message');
  const log = document.getElementById('training-log');
  let busy = false;

  async function request(action) {
    if (busy) return;
    busy = true;
    message.textContent = '正在提交任务…';
    try {
      const response = await fetch('/api/training', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action})
      });
      const result = await response.json();
      message.textContent = result.message || (result.ok ? '完成' : '任务失败');
      if (action === 'inspect' && result.ok) {
        log.textContent = `数据路径：${result.path}\n图像：${result.images} 张\nTub：${result.tubs} 个\n容量：${result.sizeMb} MB`;
      }
    } catch (error) {
      message.textContent = `请求失败：${error.message}`;
    } finally {
      busy = false;
      await update();
    }
  }

  async function update() {
    try {
      const response = await fetch('/api/training');
      const state = await response.json();
      title.textContent = state.running
        ? `训练任务：${state.kind === 'pilot' ? '基础 Pilot' : '角速度残差'}运行中`
        : `训练任务：${state.exitCode === 0 ? '最近任务已完成' : '空闲'}`;
      if (state.log && state.log.length) {
        log.textContent = state.log.join('\n');
        log.scrollTop = log.scrollHeight;
      }
      if (state.dataset) {
        document.getElementById('dataset').textContent =
          `${state.dataset.images} 张图像 · ${state.dataset.sizeMb} MB`;
      }
      document.querySelectorAll('[data-task]').forEach(button => {
        const action = button.dataset.task;
        button.disabled = state.running ? action !== 'stop' : action === 'stop';
      });
    } catch (error) {
      title.textContent = '训练任务：状态连接失败';
    }
  }

  document.querySelectorAll('[data-task]').forEach(button => {
    button.addEventListener('click', () => request(button.dataset.task));
  });
  document.addEventListener('keydown', event => {
    if (event.ctrlKey && event.key === '1') { event.preventDefault(); request('inspect'); }
    if (event.ctrlKey && event.key === '2') { event.preventDefault(); request('pilot'); }
    if (event.ctrlKey && event.key === '3') { event.preventDefault(); request('residual'); }
    if (event.key === 'Escape') request('stop');
  });
  update();
  setInterval(update, 1500);
})();
