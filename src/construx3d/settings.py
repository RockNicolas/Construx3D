from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
EXPORT_DIR = DATA_DIR / "exports"
APP_CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", str(DATA_DIR))) / "Construx3D"
MODELS_DIR = APP_CACHE_DIR / "models"
LATEST_JSON = EXPORT_DIR / "latest_scene.json"
SETTINGS_PATH = CONFIG_DIR / "gesture_settings.json"
LEGACY_SETTINGS_PATH = PROJECT_ROOT / "gesture_settings.json"
LEGACY_EXPORT_DIR = PROJECT_ROOT / "exports"
LEGACY_LATEST_JSON = LEGACY_EXPORT_DIR / "latest_scene.json"
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
WINDOW_NAME = "Construx3D - Editor por Gestos"
SETTINGS_DISPLAY_PATH = "config/gesture_settings.json"
PRIMITIVES = {1: "cube", 2: "pyramid", 3: "prism"}
PRIMITIVE_LABELS = {
    "cube": "Cubo",
    "pyramid": "Piramide",
    "prism": "Prisma",
}
PRIMITIVE_COLORS = {
    "cube": (245, 120, 120),
    "pyramid": (120, 220, 255),
    "prism": (170, 255, 120),
}


@dataclass
class CameraSettings:
    width: int = 1280
    height: int = 720


@dataclass
class TrackingSettings:
    model_complexity: int = 0
    max_num_hands: int = 2
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.55
    pinch_distance_threshold_px: float = 45.0


@dataclass
class HoldSettings:
    hold_time: float
    cooldown: float


@dataclass
class GestureSettings:
    create: HoldSettings
    duplicate: HoldSettings
    delete: HoldSettings
    undo: HoldSettings
    commit: HoldSettings
    cancel_preview: HoldSettings


@dataclass
class ZoomSettings:
    min_zoom: float = 350.0
    max_zoom: float = 1500.0


@dataclass
class SelectionSettings:
    max_cursor_distance_px: float = 120.0


@dataclass
class AppSettings:
    camera: CameraSettings
    tracking: TrackingSettings
    gestures: GestureSettings
    zoom: ZoomSettings
    selection: SelectionSettings


DEFAULT_SETTINGS = AppSettings(
    camera=CameraSettings(),
    tracking=TrackingSettings(),
    gestures=GestureSettings(
        create=HoldSettings(hold_time=0.45, cooldown=0.65),
        duplicate=HoldSettings(hold_time=0.45, cooldown=0.65),
        delete=HoldSettings(hold_time=0.45, cooldown=0.65),
        undo=HoldSettings(hold_time=0.55, cooldown=0.8),
        commit=HoldSettings(hold_time=0.35, cooldown=0.45),
        cancel_preview=HoldSettings(hold_time=0.45, cooldown=0.45),
    ),
    zoom=ZoomSettings(),
    selection=SelectionSettings(),
)


def ensure_runtime_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_hand_landmarker_model() -> Path:
    ensure_runtime_dirs()
    if HAND_LANDMARKER_MODEL_PATH.exists():
        return HAND_LANDMARKER_MODEL_PATH

    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel baixar o modelo do MediaPipe Hand Landmarker. "
            f"Baixe manualmente em {HAND_LANDMARKER_MODEL_URL} e salve em {HAND_LANDMARKER_MODEL_PATH}."
        ) from exc

    return HAND_LANDMARKER_MODEL_PATH


def latest_scene_input_path() -> Path:
    if LATEST_JSON.exists():
        return LATEST_JSON
    if LEGACY_LATEST_JSON.exists():
        return LEGACY_LATEST_JSON
    return LATEST_JSON


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_settings(data: Dict[str, Any]) -> AppSettings:
    camera = data.get("camera", {})
    tracking = data.get("tracking", {})
    gestures = data.get("gestures", {})
    zoom = data.get("zoom", {})
    selection = data.get("selection", {})

    return AppSettings(
        camera=CameraSettings(**camera),
        tracking=TrackingSettings(**tracking),
        gestures=GestureSettings(
            create=HoldSettings(**gestures.get("create", {})),
            duplicate=HoldSettings(**gestures.get("duplicate", {})),
            delete=HoldSettings(**gestures.get("delete", {})),
            undo=HoldSettings(**gestures.get("undo", {})),
            commit=HoldSettings(**gestures.get("commit", {})),
            cancel_preview=HoldSettings(**gestures.get("cancel_preview", {})),
        ),
        zoom=ZoomSettings(**zoom),
        selection=SelectionSettings(**selection),
    )


def load_settings(path: Path | None = None) -> AppSettings:
    ensure_runtime_dirs()
    active_path = path or SETTINGS_PATH
    if not active_path.exists() and LEGACY_SETTINGS_PATH.exists():
        active_path = LEGACY_SETTINGS_PATH

    defaults = asdict(DEFAULT_SETTINGS)
    if not active_path.exists():
        return DEFAULT_SETTINGS

    try:
        loaded = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS

    if not isinstance(loaded, dict):
        return DEFAULT_SETTINGS

    merged = _deep_merge(defaults, loaded)
    try:
        return _build_settings(merged)
    except TypeError:
        return DEFAULT_SETTINGS
