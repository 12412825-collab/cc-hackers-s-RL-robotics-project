(() => {
  const phrases = [
    ['一键自动训练会先执行质量检查，通过后依次训练基础 Pilot 和角速度残差；任何一步失败都会停止。', 'Automatic training first validates the dataset, then trains the base Pilot and angular-velocity residual; any failure stops the pipeline.'],
    ['请启动 Webots 世界和 donkey_webots 控制器', 'Start the Webots world and donkey_webots controller'],
    ['当前仍是旧后端，请重启 donkey_webots 控制器后再试', 'The old backend is still running. Restart the donkey_webots controller and try again.'],
    ['所有操作均在后台运行，无需终端。', 'All jobs run in the background; no terminal is required.'],
    ['重置请求已发送，但尚未收到 Webots 确认', 'Reset request sent, but Webots has not confirmed it yet'],
    ['Webots 未在 2 秒内确认重置，请确认仿真正在运行', 'Webots did not confirm the reset within 2 seconds. Check that the simulation is running.'],
    ['Webots 已确认重置，位置应回到 X≈-3、Z≈0', 'Webots confirmed the reset; position should return to X≈-3, Z≈0'],
    ['控制台已刷新并重新连接', 'Console refreshed and reconnected'],
    ['车载视频加载失败，请检查 /video 摄像头流', 'Onboard video failed to load. Check the /video camera stream.'],
    ['正在检查数据并准备自动训练…', 'Inspecting data and preparing automatic training…'],
    ['数据质量未通过，已阻止训练', 'Dataset validation failed; training was blocked'],
    ['自动训练流水线已启动', 'Automatic training pipeline started'],
    ['训练任务已启动', 'Training job started'],
    ['已有训练任务正在运行', 'A training job is already running'],
    ['当前没有运行中的训练任务', 'No training job is currently running'],
    ['尚未找到 data 数据目录', 'The data directory was not found'],
    ['未知仿真操作', 'Unknown simulation action'],
    ['未知任务', 'Unknown job'],
    ['请先训练基础 Pilot 模型', 'Train a base Pilot model first'],
    ['残差训练需要安装 PyTorch', 'Residual training requires PyTorch'],
    ['全部训练任务完成', 'All training jobs completed'],
    ['自动流水线停止：', 'Automatic pipeline stopped: '],
    ['正在停止训练', 'Stopping training'],
    ['用户请求停止训练', 'User requested training stop'],
    ['正在提交任务…', 'Submitting job…'],
    ['等待仿真数据', 'Waiting for simulation data'],
    ['连接断开，正在重连', 'Disconnected; reconnecting'],
    ['连接异常，正在重试', 'Connection error; retrying'],
    ['正在加载车载视频…', 'Loading onboard video…'],
    ['车载视频连接成功', 'Onboard video connected'],
    ['车载视角已加载', 'Onboard camera loaded'],
    ['已切换到上帝视角', 'Switched to overview'],
    ['正在重置车辆…', 'Resetting vehicle…'],
    ['Webots 已确认车辆重置', 'Webots confirmed vehicle reset'],
    ['后端返回了非 JSON 响应', 'Backend returned a non-JSON response'],
    ['训练任务：状态连接失败', 'Training Job: status connection failed'],
    ['最近流水线已完成', 'Latest pipeline completed'],
    ['后续自动任务：', 'Queued automatic jobs: '],
    ['训练任务：空闲', 'Training Job: Idle'],
    ['训练任务：', 'Training Job: '],
    ['等待训练任务…', 'Waiting for a training job…'],
    ['任务失败，退出代码', 'Job failed with exit code'],
    ['任务退出，代码', 'Job exited with code'],
    ['任务失败', 'Job failed'],
    ['请求失败：', 'Request failed: '],
    ['重置失败：', 'Reset failed: '],
    ['重置失败', 'Reset failed'],
    ['任务完成', 'Job completed'],
    ['角速度残差', 'Angular Velocity Residual'],
    ['基础 Pilot', 'Base Pilot'],
    ['训练完成', ' training completed'],
    ['运行中', ' running'],
    ['启动：', 'Started: '],
    ['自动继续：', 'Continuing automatically: '],
    ['阻止训练：', 'Training blocked: '],
    ['数据检查：', 'Dataset inspection: '],
    ['数据路径：', 'Data path: '],
    ['重复比例：', 'Duplicate ratio: '],
    ['唯一图像', 'unique images'],
    ['未通过，训练已锁定', 'failed; training is locked'],
    ['通过，可以训练', 'passed; ready to train'],
    ['尚未检查', 'Not inspected'],
    ['未通过', 'Failed'],
    ['可训练', 'Trainable'],
    ['结论：', 'Result: '],
    ['容量：', 'Size: '],
    ['记录：', 'Records: '],
    ['图像：', 'Images: '],
    ['个 catalog', 'catalogs'],
    ['个 Tub', 'Tubs'],
    ['张唯一图像', 'unique images'],
    ['张图像', 'images'],
    ['条记录', 'records'],
    ['条 ·', 'records ·'],
    ['张，唯一', 'images, unique'],
    ['项', 'items'],
    ['仿真模拟训练', 'Simulation Training'],
    ['正在连接', 'Connecting'],
    ['实时连接', 'Live'],
    ['上帝视角', 'Overview'],
    ['车载视角', 'Onboard Camera'],
    ['仿真场景俯视图', 'Simulation overview'],
    ['机器人摄像头', 'Robot camera'],
    ['刷新控制台', 'Refresh Console'],
    ['重置车辆', 'Reset Vehicle'],
    ['车辆控制', 'Vehicle Control'],
    ['方向键 / WASD', 'Arrow Keys / WASD'],
    ['向上前进 · 左右转向 · 松手停车', 'Push up to drive · Move left/right to steer · Release to stop'],
    ['前进 · ↓ 后退 · ← → 转向 · 空格急停', 'Forward · ↓ Reverse · ← → Steer · Space to stop'],
    ['空格急停', 'Space for emergency stop'],
    ['线速度指令', 'Linear Velocity Command'],
    ['角速度指令', 'Angular Velocity Command'],
    ['开始录制', 'Start Recording'],
    ['停止录制', 'Stop Recording'],
    ['正在录制', 'Recording'],
    ['未录制', 'Not Recording'],
    ['实时遥测', 'Live Telemetry'],
    ['实际速度', 'Actual Speed'],
    ['横向误差 CTE', 'Cross-track Error CTE'],
    ['障碍距离', 'Obstacle Distance'],
    ['ω 残差', 'ω Residual'],
    ['底盘与传感器', 'Chassis & Sensors'],
    ['传感器在线', 'Sensors Online'],
    ['等待数据', 'Waiting for data'],
    ['加速度 XYZ', 'Acceleration XYZ'],
    ['角速度 XYZ', 'Angular Rate XYZ'],
    ['位置 X / Z', 'Position X / Z'],
    ['左轮', 'Left Wheels'],
    ['右轮', 'Right Wheels'],
    ['连接', 'Link'],
    ['训练状态', 'Training Status'],
    ['方案 A · 仅修正 ω', 'Scheme A · Correct ω only'],
    ['差速轮速', 'Differential Wheel Speeds'],
    ['基础 ω', 'Base ω'],
    ['残差 ω', 'Residual ω'],
    ['图像', 'Image'],
    ['数据集', 'Dataset'],
    ['残差策略', 'Residual Policy'],
    ['安全状态', 'Safety'],
    ['已启用', 'Enabled'],
    ['待训练 / 未启用', 'Untrained / Disabled'],
    ['一键自动训练', 'Automatic Training'],
    ['检查数据', 'Inspect Data'],
    ['仅训练 Pilot', 'Train Pilot Only'],
    ['仅训练 ω 残差', 'Train ω Residual Only'],
    ['停止任务', 'Stop Task'],
    ['前方障碍', 'Obstacle Ahead'],
    ['偏离赛道', 'Off Track'],
    ['正常', 'Normal'],
    ['完成', 'Completed'],
    ['空闲', 'Idle'],
    ['全自动', 'Full Auto'],
    ['自动转向', 'Auto Steering'],
    ['手动', 'Manual'],
    ['急停', 'Emergency Stop'],
    ['障碍物', 'Obstacle'],
    ['目标线', 'Target Line'],
    ['轨迹', 'Trail'],
    ['车辆', 'Robot'],
    ['轮径', 'Wheel diameter'],
    ['轮距', 'Wheel separation'],
    ['起点', 'Start'],
    ['障碍', 'Obstacle']
  ].sort((a, b) => b[0].length - a[0].length);

  const language = localStorage.getItem('diffdrive-language') || 'zh';
  let applying = false;

  function translate(value) {
    if (language !== 'en' || !value) return value;
    // Telemetry updates many numeric text nodes every second. Avoid scanning
    // the full phrase table unless the value can actually contain Chinese.
    if (!/[\u3400-\u9fff]/u.test(value)) return value;
    let result = value;
    for (const [zh, en] of phrases) result = result.replaceAll(zh, en);
    return result;
  }

  function translateTextNode(node) {
    const current = node.nodeValue;
    const translated = translate(current);
    // Writing the same value still emits a characterData mutation in some
    // browsers, which can otherwise make the observer loop continuously.
    if (translated !== current) node.nodeValue = translated;
  }

  function translateElement(element) {
    if (!element || element.nodeType !== Node.ELEMENT_NODE) return;
    for (const attribute of ['alt', 'aria-label', 'title', 'placeholder']) {
      if (!element.hasAttribute(attribute)) continue;
      const current = element.getAttribute(attribute);
      const translated = translate(current);
      if (translated !== current) element.setAttribute(attribute, translated);
    }
  }

  function translateTree(root) {
    if (language !== 'en') return;
    if (root.nodeType === Node.TEXT_NODE) {
      if (!['SCRIPT', 'STYLE'].includes(root.parentElement?.tagName)) translateTextNode(root);
      return;
    }
    translateElement(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.nodeType === Node.TEXT_NODE) {
        if (!['SCRIPT', 'STYLE'].includes(node.parentElement?.tagName)) translateTextNode(node);
      } else translateElement(node);
    }
  }

  function apply() {
    applying = true;
    translateTree(document.body);
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    document.title = language === 'en' ? 'Simulation Training' : '仿真模拟训练';
    document.getElementById('language-toggle').textContent = language === 'en' ? '中文' : 'EN';
    applying = false;
  }

  new MutationObserver(mutations => {
    if (applying || language !== 'en') return;
    applying = true;
    for (const mutation of mutations) {
      if (mutation.type === 'characterData') translateTree(mutation.target);
      mutation.addedNodes.forEach(translateTree);
    }
    applying = false;
  }).observe(document.body, {subtree: true, childList: true, characterData: true});

  document.getElementById('language-toggle').addEventListener('click', () => {
    localStorage.setItem('diffdrive-language', language === 'zh' ? 'en' : 'zh');
    location.reload();
  });
  apply();
})();
