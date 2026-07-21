(() => {
  const title = document.getElementById('job-title');
  const message = document.getElementById('job-message');
  const log = document.getElementById('training-log');
  let busy = false;

  function datasetText(data) {
    if (!data) return '尚未检查';
    const verdict = data.trainable ? '可训练' : `未通过（${data.issues.length} 项）`;
    return `${data.records} 条 · 唯一图像 ${data.uniqueImages}/${data.images} · ${verdict}`;
  }

  function inspectionText(data) {
    const lines = [
      `数据路径：${data.path}`,
      `记录：${data.records} 条（${data.catalogs} 个 catalog，${data.tubs} 个 Tub）`,
      `图像：${data.images} 张，唯一 ${data.uniqueImages} 张`,
      `重复比例：${(data.duplicateRatio * 100).toFixed(1)}%`,
      `容量：${data.sizeMb} MB`,
      `结论：${data.trainable ? '通过，可以训练' : '未通过，训练已锁定'}`
    ];
    if (data.issues.length) lines.push('', ...data.issues.map(issue => `• ${issue}`));
    return lines.join('\n');
  }

  async function request(action) {
    if (busy) return;
    busy = true;
    message.textContent = action === 'auto' ? '正在检查数据并准备自动训练…' : '正在提交任务…';
    try {
      const response = await fetch('/api/training', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action})
      });
      const result = await response.json();
      message.textContent = result.message || (result.ok ? '完成' : '任务失败');
      if (action === 'inspect' && result.ok) log.textContent = inspectionText(result);
      if (result.dataset && !result.ok) log.textContent = inspectionText(result.dataset);
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
      const name = state.kind === 'pilot' ? '基础 Pilot' : '角速度残差';
      title.textContent = state.running ? `训练任务：${name}运行中` :
        `训练任务：${state.exitCode === 0 ? '最近流水线已完成' : '空闲'}`;
      if (state.pipeline?.length) message.textContent = `后续自动任务：${state.pipeline.join(' → ')}`;
      if (state.log?.length) {
        log.textContent = state.log.join('\n');
        log.scrollTop = log.scrollHeight;
      }
      document.getElementById('dataset').textContent = datasetText(state.dataset);
      document.querySelectorAll('[data-task]').forEach(button => {
        const action = button.dataset.task;
        button.disabled = state.running ? action !== 'stop' : action === 'stop';
      });
    } catch (error) {
      title.textContent = '训练任务：状态连接失败';
    }
  }

  document.querySelectorAll('[data-task]').forEach(button =>
    button.addEventListener('click', () => request(button.dataset.task)));
  document.addEventListener('keydown', event => {
    if (event.ctrlKey && event.key === '0') { event.preventDefault(); request('auto'); }
    if (event.ctrlKey && event.key === '1') { event.preventDefault(); request('inspect'); }
    if (event.ctrlKey && event.key === '2') { event.preventDefault(); request('pilot'); }
    if (event.ctrlKey && event.key === '3') { event.preventDefault(); request('residual'); }
    if (event.key === 'Escape') request('stop');
  });
  update();
  setInterval(update, 1500);
})();
