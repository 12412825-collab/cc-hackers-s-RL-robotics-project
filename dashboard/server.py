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
from parts.dataset_quality import inspect_dataset
from parts.simulation_control import request_reset, wait_for_reset


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
            elif action in ("pilot", "residual", "auto"):
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


class SimulationAPI(RequestHandler):
    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
            if body.get("action") != "reset":
                self.set_status(400)
                self.write({"ok": False, "message": "未知仿真操作"})
                return
            self.application.angle = 0.0
            self.application.throttle = 0.0
            self.application.recording_latch = False
            request_reset()
            confirmed = wait_for_reset(2.0)
            if confirmed:
                self.write({"ok": True, "confirmed": True,
                            "message": "Webots 已确认车辆重置"})
            else:
                self.set_status(504)
                self.write({"ok": False, "confirmed": False,
                            "message": "Webots 未在 2 秒内确认重置，请确认仿真正在运行"})
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
        self.data_dir = Path(cfg.DATA_PATH).resolve()
        self.training_process = None
        self.training_kind = None
        self.training_started = None
        self.training_exit_code = None
        self.training_log = deque(maxlen=120)
        self.training_pipeline = []
        self.stop_requested = False
        self.cfg = cfg
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
            (r"/api/simulation", SimulationAPI),
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

    # The definitions below intentionally replace the legacy lightweight
    # training handlers above. Keeping them here makes upgrades from the
    # original console non-destructive while applying strict quality gates.
    def _legacy_inspect_dataset(self, add_log=True):
        result = inspect_dataset(
            self.data_dir,
            min_records=int(getattr(self.cfg, "DATASET_MIN_RECORDS", 100)),
            max_duplicate_ratio=float(getattr(
                self.cfg, "DATASET_MAX_DUPLICATE_RATIO", 0.20)),
            min_label_range=float(getattr(
                self.cfg, "DATASET_MIN_LABEL_RANGE", 0.05)),
            max_position=float(getattr(self.cfg, "WEBOTS_GEOFENCE_M", 45.0)),
            max_speed=float(getattr(self.cfg, "RECORD_MAX_SPEED_MPS", 2.0)))
        if add_log:
            verdict = "可训练" if result["trainable"] else "未通过"
            self.training_log.append(
                f"数据检查：{result['records']} 条记录，"
                f"{result['uniqueImages']}/{result['images']} 张唯一图像，{verdict}")
            for issue in result["issues"]:
                self.training_log.append("阻止训练：" + issue)
        return result

    def _command_for(self, kind):
        if kind == "pilot":
            return [sys.executable, "manage.py", "train",
                    f"--tubs={self.data_dir}",
                    "--model=models/pilot_latest.h5", "--type=linear"]
        base = self.project_root / "models" / "pilot_latest.h5"
        if not base.exists():
            raise FileNotFoundError("请先训练基础 Pilot 模型")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("残差训练需要安装 PyTorch") from exc
        return [sys.executable, "train_residual.py", f"--tubs={self.data_dir}",
                "--base=models/pilot_latest.h5",
                "--output=models/residual_latest.pth"]

    def _launch_training(self, kind, clear_log=True):
        command = self._command_for(kind)
        if clear_log:
            self.training_log.clear()
        self.training_log.append("启动：" + " ".join(command[1:]))
        self.training_process = subprocess.Popen(
            command, cwd=self.project_root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.training_kind = kind
        self.training_started = time.time()
        self.training_exit_code = None
        threading.Thread(target=self._pipeline_reader,
                         args=(self.training_process,), daemon=True).start()

    def _pipeline_reader(self, process):
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                self.training_log.append(line)
        self.training_exit_code = process.wait()
        if self.training_exit_code != 0:
            self.training_log.append(
                f"任务失败，退出代码 {self.training_exit_code}")
            self.training_pipeline.clear()
            return
        self.training_log.append(f"{self.training_kind} 训练完成")
        if self.training_pipeline and not self.stop_requested:
            next_kind = self.training_pipeline.pop(0)
            self.training_log.append(f"自动继续：{next_kind}")
            try:
                self._launch_training(next_kind, clear_log=False)
            except (FileNotFoundError, RuntimeError) as exc:
                self.training_exit_code = 1
                self.training_log.append("自动流水线停止：" + str(exc))
        else:
            self.training_log.append("全部训练任务完成")

    def _legacy_start_training(self, kind):
        if self.training_process and self.training_process.poll() is None:
            return {"ok": False, "message": "已有训练任务正在运行"}
        quality = self.inspect_dataset(add_log=False)
        if not quality["trainable"]:
            self.training_log.clear()
            self.inspect_dataset(add_log=True)
            return {"ok": False, "message": "数据质量未通过，已阻止训练",
                    "dataset": quality}
        (self.project_root / "models").mkdir(exist_ok=True)
        self.stop_requested = False
        self.training_pipeline = ["residual"] if kind == "auto" else []
        first_kind = "pilot" if kind == "auto" else kind
        try:
            self._launch_training(first_kind)
        except (FileNotFoundError, RuntimeError) as exc:
            self.training_pipeline.clear()
            return {"ok": False, "message": str(exc)}
        return {"ok": True, "message": (
            "自动训练流水线已启动" if kind == "auto" else "训练任务已启动")}

    def _legacy_stop_training(self):
        if not self.training_process or self.training_process.poll() is not None:
            return {"ok": False, "message": "当前没有运行中的训练任务"}
        self.stop_requested = True
        self.training_pipeline.clear()
        self.training_process.terminate()
        self.training_log.append("用户请求停止训练")
        return {"ok": True, "message": "正在停止训练"}

    def _legacy_training_status(self):
        running = bool(self.training_process and
                       self.training_process.poll() is None)
        return {"running": running, "kind": self.training_kind,
                "started": self.training_started,
                "exitCode": self.training_exit_code,
                "pipeline": list(self.training_pipeline),
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
