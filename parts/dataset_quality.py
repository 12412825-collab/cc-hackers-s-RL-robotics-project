"""Training-data validation shared by recording and the web console."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class RecordingQualityGate:
    """Allow recording only while commands and simulator state are plausible."""

    def __init__(self, cfg):
        self.min_command = float(getattr(cfg, "RECORD_MIN_COMMAND", 0.02))
        self.max_position = float(getattr(cfg, "WEBOTS_GEOFENCE_M", 45.0))
        self.max_speed = float(getattr(cfg, "RECORD_MAX_SPEED_MPS", 2.0))
        self.max_duplicate_frames = int(
            getattr(cfg, "RECORD_MAX_DUPLICATE_FRAMES", 3))
        self.last_digest = None
        self.duplicate_frames = 0
        self.reason = "waiting"

    def run(self, recording, image, angle, throttle, linear, angular,
            speed, pos_x, pos_y, pos_z):
        if not recording:
            self.reason = "recording disabled"
            return False
        values = (angle, throttle, linear, angular, speed, pos_x, pos_y, pos_z)
        if not all(_finite(value) for value in values):
            self.reason = "non-finite telemetry"
            return False
        if max(abs(float(pos_x)), abs(float(pos_z))) > self.max_position:
            self.reason = "position outside geofence"
            return False
        if abs(float(speed)) > self.max_speed:
            self.reason = "implausible speed"
            return False
        moving_command = max(abs(float(angle)), abs(float(throttle)),
                             abs(float(linear)), abs(float(angular)))
        if moving_command < self.min_command:
            self.reason = "zero command"
            return False
        if image is None:
            self.reason = "missing camera image"
            return False
        try:
            digest = hashlib.blake2b(memoryview(image), digest_size=8).digest()
        except TypeError:
            digest = hashlib.blake2b(bytes(image), digest_size=8).digest()
        self.duplicate_frames = (self.duplicate_frames + 1
                                 if digest == self.last_digest else 0)
        self.last_digest = digest
        if self.duplicate_frames > self.max_duplicate_frames:
            self.reason = "camera frame is frozen"
            return False
        self.reason = "ok"
        return True


def inspect_dataset(data_dir, *, min_records=100, max_duplicate_ratio=0.20,
                    min_label_range=0.05, max_position=45.0,
                    max_speed=2.0):
    """Inspect a DonkeyCar Tub v2 directory and return JSON-safe metrics."""
    root = Path(data_dir)
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    manifests = list(root.rglob("manifest.json")) if root.exists() else []
    catalogs = sorted(root.rglob("catalog_*.catalog")) if root.exists() else []
    rows, bad_rows = [], 0
    for catalog in catalogs:
        with catalog.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    bad_rows += 1

    image_names = {path.name: path for path in images}
    referenced = [row.get("cam/image_array") for row in rows]
    missing_images = sum(name not in image_names for name in referenced)
    hashes = Counter()
    for path in images:
        try:
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()] += 1
        except OSError:
            pass
    duplicate_images = sum(hashes.values()) - len(hashes)
    duplicate_ratio = duplicate_images / len(images) if images else 1.0

    def stats(field):
        values = [float(row[field]) for row in rows
                  if field in row and _finite(row[field])]
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "range": max(values) - min(values) if values else 0.0,
            "zeroRatio": (sum(value == 0 for value in values) / len(values)
                          if values else 1.0),
        }

    fields = {name: stats(name) for name in (
        "user/angle", "user/throttle", "linear/velocity",
        "angular/velocity", "enc/speed", "pos/speed", "pos/cte")}
    positions = [abs(float(row[key])) for row in rows
                 for key in ("pos/pos_x", "pos/pos_z")
                 if key in row and _finite(row[key])]
    speeds = [abs(float(row["pos/speed"])) for row in rows
              if "pos/speed" in row and _finite(row["pos/speed"])]

    issues = []
    if len(rows) < min_records:
        issues.append(f"样本不足：{len(rows)} < {min_records}")
    if bad_rows:
        issues.append(f"存在 {bad_rows} 条损坏记录")
    if missing_images:
        issues.append(f"缺少 {missing_images} 张引用图像")
    if duplicate_ratio > max_duplicate_ratio:
        issues.append(f"重复图像比例过高：{duplicate_ratio:.1%}")
    for label in ("user/angle", "user/throttle"):
        if fields[label]["range"] < min_label_range:
            issues.append(f"{label} 缺少变化")
    if positions and max(positions) > max_position:
        issues.append(f"位置越界：最大绝对坐标 {max(positions):.2f} m")
    if speeds and max(speeds) > max_speed:
        issues.append(f"速度异常：最大 {max(speeds):.2f} m/s")

    return {
        "ok": True,
        "trainable": not issues,
        "path": str(root.resolve()),
        "records": len(rows),
        "images": len(images),
        "uniqueImages": len(hashes),
        "duplicateRatio": round(duplicate_ratio, 4),
        "missingImages": missing_images,
        "badRecords": bad_rows,
        "tubs": len(manifests),
        "catalogs": len(catalogs),
        "sizeMb": round(sum(path.stat().st_size for path in files) / 1048576, 1),
        "fields": fields,
        "issues": issues,
    }
