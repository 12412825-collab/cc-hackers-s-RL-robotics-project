(() => {
  const pairs = [
    ['仿真模拟训练', 'Simulation Training'],
    ['正在连接', 'Connecting'], ['实时连接', 'Live'], ['连接断开', 'Disconnected'],
    ['上帝视角', 'Overview'], ['车载视角', 'Onboard Camera'],
    ['方向键 / WASD', 'Arrow Keys / WASD'],
    ['↑ 前进　↓ 后退　← → 转向　空格急停', '↑ Forward  ↓ Reverse  ← → Turn  Space E-stop'],
    ['手动', 'Manual'], ['自动转向', 'Auto Steering'], ['全自动', 'Full Auto'],
    ['未录制', 'Not Recording'], ['● 正在录制', '● Recording'],
    ['车辆控制', 'Vehicle Control'], ['方向键 / WASD · 空格急停', 'Arrow Keys / WASD · Space E-stop'],
    ['向上前进 · 左右转向 · 松手停车', 'Up to drive · Left/right to turn · Release to stop'],
    ['线速度指令', 'Linear Velocity Command'], ['角速度指令', 'Angular Velocity Command'],
    ['急停', 'Emergency Stop'], ['开始录制', 'Start Recording'], ['停止录制', 'Stop Recording'],
    ['实时遥测', 'Live Telemetry'], ['实际速度', 'Actual Speed'],
    ['横向误差 CTE', 'Cross-track Error CTE'], ['障碍距离', 'Obstacle Distance'],
    ['ω 残差', 'ω Residual'], ['底盘与传感器', 'Chassis & Sensors'],
    ['左轮', 'Left Wheels'], ['右轮', 'Right Wheels'], ['位置 X / Z', 'Position X / Z'],
    ['加速度 XYZ', 'Acceleration XYZ'], ['角速度 XYZ', 'Angular Rate XYZ'],
    ['轮径', 'Wheel Diameter'], ['轮距', 'Track Width'],
    ['训练状态', 'Training Status'], ['方案 A · 仅修正 ω', 'Plan A · Correct ω Only'],
    ['图像', 'Image'], ['基础 ω', 'Base ω'], ['残差 ω', 'Residual ω'], ['差速轮速', 'Differential Wheel Speed'],
    ['数据集', 'Dataset'], ['残差策略', 'Residual Policy'], ['安全状态', 'Safety'],
    ['已启用', 'Enabled'], ['待训练 / 未启用', 'Not Trained / Disabled'], ['正常', 'Normal'],
    ['检查数据', 'Inspect Data'], ['训练 Pilot', 'Train Pilot'], ['训练 ω 残差', 'Train ω Residual'], ['停止任务', 'Stop Task'],
    ['训练任务：空闲', 'Training Job: Idle'], ['训练任务：最近任务已完成', 'Training Job: Last Job Completed'],
    ['所有操作均在后台运行，不需要终端。', 'All jobs run in the background; no terminal is required.'],
    ['等待训练任务…', 'Waiting for a training job…'],
    ['建议顺序：检查数据 → 训练基础 Pilot → Webots 验证 → 训练角速度残差。', 'Recommended: inspect data → train Pilot → validate in Webots → train angular residual.']
  ];
  const zhToEn = new Map(pairs), enToZh = new Map(pairs.map(([zh, en]) => [en, zh]));
  let language = localStorage.getItem('diffdrive-language') || 'zh';
  let translating = false;

  function dynamic(text, target) {
    if (target === 'en') {
      return text
        .replace(/^(\d+) 条记录$/, '$1 records')
        .replace(/^(\d+) 张图像 · ([\d.]+) MB$/, '$1 images · $2 MB')
        .replace(/^训练任务：基础 Pilot运行中$/, 'Training Job: Pilot Running')
        .replace(/^训练任务：角速度残差运行中$/, 'Training Job: Angular Residual Running');
    }
    return text
      .replace(/^(\d+) records$/, '$1 条记录')
      .replace(/^(\d+) samples$/, '$1 条记录')
      .replace(/^(\d+) images · ([\d.]+) MB$/, '$1 张图像 · $2 MB')
      .replace(/^Training Job: Pilot Running$/, '训练任务：基础 Pilot运行中')
      .replace(/^Training Job: Angular Residual Running$/, '训练任务：角速度残差运行中');
  }

  function translateTextNode(node, target) {
    const raw = node.nodeValue;
    const trimmed = raw.trim();
    if (!trimmed) return;
    const table = target === 'en' ? zhToEn : enToZh;
    const translated = table.get(trimmed) || dynamic(trimmed, target);
    if (translated !== trimmed) node.nodeValue = raw.replace(trimmed, translated);
  }

  function apply(target) {
    translating = true;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) translateTextNode(node, target);
    document.documentElement.lang = target === 'en' ? 'en' : 'zh-CN';
    document.title = target === 'en' ? 'Simulation Training' : '仿真模拟训练';
    const button = document.getElementById('language-toggle');
    button.textContent = target === 'en' ? '中文' : 'EN';
    button.setAttribute('aria-label', target === 'en' ? '切换为中文' : 'Switch to English');
    translating = false;
  }

  new MutationObserver(mutations => {
    if (translating) return;
    translating = true;
    for (const mutation of mutations) {
      if (mutation.type === 'characterData') translateTextNode(mutation.target, language);
      mutation.addedNodes.forEach(node => {
        if (node.nodeType === Node.TEXT_NODE) translateTextNode(node, language);
      });
    }
    translating = false;
  }).observe(document.body, {subtree: true, childList: true, characterData: true});

  document.getElementById('language-toggle').addEventListener('click', () => {
    language = language === 'zh' ? 'en' : 'zh';
    localStorage.setItem('diffdrive-language', language);
    apply(language);
  });
  apply(language);
})();
