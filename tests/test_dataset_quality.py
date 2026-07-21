import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parts.dataset_quality import RecordingQualityGate, inspect_dataset


def config():
    return SimpleNamespace(
        RECORD_MIN_COMMAND=0.02,
        RECORD_MAX_DUPLICATE_FRAMES=1,
        RECORD_MAX_SPEED_MPS=2.0,
        WEBOTS_GEOFENCE_M=45.0,
    )


def test_recording_gate_rejects_stationary_and_frozen_frames():
    gate = RecordingQualityGate(config())
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert not gate.run(True, image, 0, 0, 0, 0, 0, 0, 0.03, 0)
    assert gate.reason == "zero command"
    assert gate.run(True, image, 0.2, 0.3, 0.1, 0.1, 0.1, 0, 0.03, 0)
    assert gate.run(True, image, 0.2, 0.3, 0.1, 0.1, 0.1, 0, 0.03, 0)
    assert not gate.run(True, image, 0.2, 0.3, 0.1, 0.1, 0.1, 0, 0.03, 0)
    assert gate.reason == "camera frame is frozen"


def test_recording_gate_rejects_implausible_state():
    gate = RecordingQualityGate(config())
    image = np.ones((4, 4, 3), dtype=np.uint8)
    assert not gate.run(True, image, 0.2, 0.3, 0.1, 0.1, 3.0, 0, 0.03, 0)
    assert gate.reason == "implausible speed"
    assert not gate.run(True, image, 0.2, 0.3, 0.1, 0.1, 0.1, 50, 0.03, 0)
    assert gate.reason == "position outside geofence"


def test_inspector_blocks_duplicate_zero_label_dataset(tmp_path):
    data = tmp_path / "data"
    images = data / "images"
    images.mkdir(parents=True)
    (data / "manifest.json").write_text("[]\n", encoding="utf-8")
    image_bytes = b"same-image"
    rows = []
    for index in range(3):
        name = f"{index}_cam_image_array_.jpg"
        (images / name).write_bytes(image_bytes)
        rows.append({
            "_index": index, "cam/image_array": name,
            "user/angle": 0.0, "user/throttle": 0.0,
            "pos/pos_x": 0.0, "pos/pos_z": 0.0, "pos/speed": 0.0,
        })
    (data / "catalog_0.catalog").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = inspect_dataset(data, min_records=2)
    assert result["records"] == 3
    assert result["uniqueImages"] == 1
    assert not result["trainable"]
    assert any("重复图像" in issue for issue in result["issues"])
    assert any("user/angle" in issue for issue in result["issues"])


def test_current_dataset_is_safely_rejected():
    data = Path(__file__).parents[1] / "data" / "current"
    if not data.exists():
        return
    result = inspect_dataset(data)
    assert not result["trainable"]
