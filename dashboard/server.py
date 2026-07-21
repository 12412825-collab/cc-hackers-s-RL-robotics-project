"""Training-focused web console compatible with DonkeyCar's drive protocol."""

import asyncio
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler, StaticFileHandler

from donkeycar.parts.web_controller.web import VideoAPI, WebSocketDriveAPI


class ConsoleHandler(RequestHandler):
    def get(self):
        self.render("console.html", config=self.application.console_config)


class TrainingAPI(RequestHandler):
    def get(self):
        self.write(self.application.training_status())

    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
            action = body.get("action")
            if action == "inspect":
                result = self.application.inspect_dataset()
            elif action in ("pilot", "residual"):
                result = self.application.start_training(action)
            elif action == "stop":
                result = self.application.stop_training()
            else:
                self.set_status(400)
                result = {"ok": False, "message": "未知任务"}
            self.write(result)
        except Exception as exc:
            self.set_status(500)
            self.write({"ok": False, "message": str(exc)})


class TrainingConsole(Application):
    """DonkeyCar controller plus live simulation/training telemetry."""

    def __init__(self, cfg):
        root = os.path.dirname(os.path.abspath(__file__))
        self.static_file_path = os.path.join(root, "static")
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = getattr(cfg, "WEB_INIT_MODE", "user")
        self.mode_latch = None
        self.recording = False
        self.recording_latch = None
        self.buttons = {}
        self.num_records = 0
        self.img_arr = None
        self.wsclients = []
        self.loop = None
        self.port = int(getattr(cfg, "WEB_CONTROL_PORT", 8887))
        self.last_publish = 0.0
        self.project_root = Path(root).parent
        self.training_process = None
        self.training_kind = None
        self.training_started = None
        self.training_exit_code = None
        self.training_log = deque(maxlen=120)
        self.console_config = {
            "maxLinear": float(cfg.MAX_LINEAR_VELOCITY),
            "maxAngular": float(cfg.MAX_ANGULAR_VELOCITY),
            "wheelRadius": float(cfg.WHEEL_RADIUS),
            "wheelSeparation": float(cfg.WHEEL_SEPARATION),
            "residualEnabled": bool(getattr(cfg, "RESIDUAL_RL", False)),
            "simulator": str(getattr(cfg, "SIMULATOR", "none")),
        }
        handlers = [
            (r"/", ConsoleHandler),
            (r"/drive", ConsoleHandler),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/video", VideoAPI),
            (r"/api/training", TrainingAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]
        super().__init__(handlers, template_path=os.path.join(root, "templates"))

    def inspect_dataset(self, add_log=True):
        data_dir = self.project_root / "data"
        images = sum(1 for suffix in ("*.jpg", "*.jpeg", "*.png")
                     for _ in data_dir.rglob(suffix)) if data_dir.exists() else 0
        manifests = list(data_dir.rglob("manifest.json")) if data_dir.exists() else []
        size = sum(path.stat().st_size for path in data_dir.rglob("*")
                   if path.is_file()) if data_dir.exists() else 0
        result = {"ok": True, "images": images, "tubs": len(manifests),
                  "sizeMb": round(size / 1048576, 1),
                  "path": str(data_dir)}
        if add_log:
            self.training_log.append(
                f"数据检查：{images} 张图像，{len(manifests)} 个 Tub，{result['sizeMb']} MB")
        return result

    def _reader(self, process):
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                self.training_log.append(line)
        self.training_exit_code = process.wait()
        self.training_log.append(
            "任务完成" if self.training_exit_code == 0
            else f"任务退出，代码 {self.training_exit_code}")

    def start_training(self, kind):
        if self.training_process and self.training_process.poll() is None:
            return {"ok": False, "message": "已有训练任务正在运行"}
        models = self.project_root / "models"
        models.mkdir(exist_ok=True)
        data = self.project_root / "data"
        if not data.exists():
            return {"ok": False, "message": "尚未找到 data 数据目录"}
        if kind == "pilot":
            command = [sys.executable, "manage.py", "train", "--tubs=data",
                       "--model=models/pilot_latest.h5", "--type=linear"]
        else:
            base = models / "pilot_latest.h5"
            if not base.exists():
                return {"ok": False, "message": "请先训练基础 Pilot 模型"}
            try:
                import torch  # noqa: F401
            except ImportError:
                return {"ok": False, "message": "残差训练需要先安装 PyTorch"}
            command = [sys.executable, "train_residual.py", "--tubs=data",
                       "--base=models/pilot_latest.h5",
                       "--output=models/residual_latest.pth"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.training_log.clear()
        self.training_log.append("启动：" + " ".join(command[1:]))
        self.training_process = subprocess.Popen(
            command, cwd=self.project_root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, creationflags=flags)
        self.training_kind = kind
        self.training_started = time.time()
        self.training_exit_code = None
        threading.Thread(target=self._reader, args=(self.training_process,),
                         daemon=True).start()
        return {"ok": True, "message": "训练任务已启动"}

    def stop_training(self):
        if not self.training_process or self.training_process.poll() is not None:
            return {"ok": False, "message": "当前没有运行中的训练任务"}
        self.training_process.terminate()
        self.training_log.append("用户请求停止训练")
        return {"ok": True, "message": "正在停止训练"}

    def training_status(self):
        running = bool(self.training_process and
                       self.training_process.poll() is None)
        return {"running": running, "kind": self.training_kind,
                "started": self.training_started,
                "exitCode": self.training_exit_code,
                "log": list(self.training_log),
                "dataset": self.inspect_dataset(add_log=False)}

    def update(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        self.loop = IOLoop.instance()
        self.loop.start()

    def _publish(self, payload):
        if not self.loop or not self.wsclients:
            return

        def send():
            message = json.dumps(payload, allow_nan=False)
            for client in list(self.wsclients):
                try:
                    client.write_message(message)
                except Exception:
                    pass

        self.loop.add_callback(send)

    @staticmethod
    def _number(value, default=0.0):
        try:
            value = float(value)
            return value if value == value and abs(value) != float("inf") else default
        except (TypeError, ValueError):
            return default

    def run_threaded(self, image=None, num_records=0, mode=None,
                     recording=None, steering=0.0, throttle=0.0,
                     linear=0.0, angular=0.0, left_speed=0.0,
                     right_speed=0.0, speed=0.0, distance=None,
                     pos_x=0.0, pos_y=0.0, pos_z=0.0, cte=0.0,
                     acl_x=0.0, acl_y=0.0, acl_z=0.0,
                     gyr_x=0.0, gyr_y=0.0, gyr_z=0.0,
                     residual=0.0):
        self.img_arr = image
        self.num_records = int(num_records or 0)
        if mode is not None:
            self.mode = mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
        if recording is not None:
            self.recording = bool(recording)
        if self.recording_latch is not None:
            self.recording = self.recording_latch
            self.recording_latch = None

        now = time.monotonic()
        if now - self.last_publish >= 0.10:
            self.last_publish = now
            self._publish({
                "driveMode": self.mode,
                "recording": self.recording,
                "num_records": self.num_records,
                "telemetry": {
                    "steering": self._number(steering),
                    "throttle": self._number(throttle),
                    "linear": self._number(linear),
                    "angular": self._number(angular),
                    "leftSpeed": self._number(left_speed),
                    "rightSpeed": self._number(right_speed),
                    "speed": self._number(speed),
                    "distance": self._number(distance, -1.0),
                    "x": self._number(pos_x), "y": self._number(pos_y),
                    "z": self._number(pos_z), "cte": self._number(cte),
                    "acl": [self._number(acl_x), self._number(acl_y),
                            self._number(acl_z)],
                    "gyr": [self._number(gyr_x), self._number(gyr_y),
                            self._number(gyr_z)],
                    "residual": self._number(residual),
                },
            })

        buttons = self.buttons
        self.buttons = {key: False for key, pressed in buttons.items() if pressed}
        return self.angle, self.throttle, self.mode, self.recording, buttons

    run = run_threaded

    def shutdown(self):
        self.angle = 0.0
        self.throttle = 0.0
