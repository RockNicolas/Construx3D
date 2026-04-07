from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .settings import PRIMITIVE_COLORS, PRIMITIVE_LABELS


@dataclass
class Shape3D:
    shape_id: int
    kind: str
    position: List[float]
    scale: float
    rotation_y: float
    color: Tuple[int, int, int]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def rotate_y(point: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = point
    return np.array([x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a], dtype=float)


def rotate_x(point: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = point
    return np.array([x, y * cos_a - z * sin_a, y * sin_a + z * cos_a], dtype=float)


def shape_geometry(kind: str, scale: float) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    half = scale / 2.0

    if kind == "cube":
        vertices = [
            np.array([x, y, z], dtype=float)
            for x in (-half, half)
            for y in (-half, half)
            for z in (-half, half)
        ]
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]
        return vertices, edges

    if kind == "pyramid":
        vertices = [
            np.array([-half, -half, -half], dtype=float),
            np.array([half, -half, -half], dtype=float),
            np.array([half, -half, half], dtype=float),
            np.array([-half, -half, half], dtype=float),
            np.array([0.0, half, 0.0], dtype=float),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4),
        ]
        return vertices, edges

    vertices = [
        np.array([-half, -half, -half], dtype=float),
        np.array([half, -half, -half], dtype=float),
        np.array([0.0, half, -half], dtype=float),
        np.array([-half, -half, half], dtype=float),
        np.array([half, -half, half], dtype=float),
        np.array([0.0, half, half], dtype=float),
    ]
    edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5),
    ]
    return vertices, edges


class GestureLatch:
    def __init__(self) -> None:
        self._starts: Dict[str, float] = {}
        self._cooldowns: Dict[str, float] = {}

    def update(self, name: str, active: bool, hold_time: float, cooldown: float, now: float) -> bool:
        if not active:
            self._starts.pop(name, None)
            return False

        if now < self._cooldowns.get(name, 0.0):
            return False

        start = self._starts.setdefault(name, now)
        if now - start >= hold_time:
            self._cooldowns[name] = now + cooldown
            self._starts[name] = now
            return True
        return False

    def progress(self, name: str, active: bool, hold_time: float, now: float) -> float:
        if not active:
            self._starts.pop(name, None)
            return 0.0

        start = self._starts.setdefault(name, now)
        return clamp((now - start) / hold_time, 0.0, 1.0)


class Scene3D:
    def __init__(self) -> None:
        self.shapes: List[Shape3D] = []
        self.preview_shape: Optional[Shape3D] = None
        self.selected_id: Optional[int] = None
        self.next_id = 1
        self.camera_distance = 7.5
        self.zoom = 700.0
        self.history: List[Dict[str, object]] = []

    def snapshot(self) -> None:
        state = {
            "shapes": copy.deepcopy([asdict(shape) for shape in self.shapes]),
            "selected_id": self.selected_id,
            "next_id": self.next_id,
            "zoom": self.zoom,
        }
        self.history.append(state)
        self.history = self.history[-30:]

    def undo(self) -> bool:
        if not self.history:
            return False

        previous = self.history.pop()
        self.shapes = [Shape3D(**shape_data) for shape_data in previous["shapes"]]
        self.selected_id = previous["selected_id"]
        self.next_id = previous["next_id"]
        self.zoom = float(previous["zoom"])
        self.preview_shape = None
        return True

    def create_preview(self, kind: str, position: List[float]) -> None:
        self.preview_shape = Shape3D(
            shape_id=self.next_id,
            kind=kind,
            position=position,
            scale=1.25,
            rotation_y=0.45,
            color=PRIMITIVE_COLORS[kind],
        )

    def commit_preview(self) -> bool:
        if self.preview_shape is None:
            return False

        self.snapshot()
        committed = copy.deepcopy(self.preview_shape)
        self.shapes.append(committed)
        self.selected_id = committed.shape_id
        self.next_id += 1
        self.preview_shape = None
        return True

    def cancel_preview(self) -> None:
        self.preview_shape = None

    def get_selected(self) -> Optional[Shape3D]:
        if self.selected_id is None:
            return None
        for shape in self.shapes:
            if shape.shape_id == self.selected_id:
                return shape
        self.selected_id = None
        return None

    def duplicate_selected_to_preview(self) -> bool:
        source = self.get_selected()
        if source is None:
            return False

        duplicate = copy.deepcopy(source)
        duplicate.shape_id = self.next_id
        duplicate.position = [source.position[0] + 0.8, source.position[1] + 0.5, source.position[2]]
        self.preview_shape = duplicate
        return True

    def delete_selected(self) -> bool:
        if self.selected_id is None:
            return False

        remaining = [shape for shape in self.shapes if shape.shape_id != self.selected_id]
        if len(remaining) == len(self.shapes):
            return False

        self.snapshot()
        self.shapes = remaining
        self.selected_id = self.shapes[-1].shape_id if self.shapes else None
        return True

    def export_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_distance": self.camera_distance,
            "zoom": self.zoom,
            "next_id": self.next_id,
            "shapes": [asdict(shape) for shape in self.shapes],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def import_json(self, path: Path) -> bool:
        if not path.exists():
            return False

        data = json.loads(path.read_text(encoding="utf-8"))
        self.shapes = [Shape3D(**shape_data) for shape_data in data.get("shapes", [])]
        self.next_id = int(data.get("next_id", len(self.shapes) + 1))
        self.zoom = float(data.get("zoom", self.zoom))
        self.camera_distance = float(data.get("camera_distance", self.camera_distance))
        self.selected_id = self.shapes[-1].shape_id if self.shapes else None
        self.preview_shape = None
        self.history.clear()
        return True

    def world_from_screen(self, point: Tuple[int, int], frame_size: Tuple[int, int]) -> List[float]:
        width, height = frame_size
        center_x = width / 2.0
        center_y = height / 2.0
        x = (point[0] - center_x) * self.camera_distance / self.zoom
        y = (center_y - point[1]) * self.camera_distance / self.zoom
        return [x, y, 0.0]

    def project(self, point: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = frame_size
        center_x = width / 2.0
        center_y = height / 2.0
        depth = point[2] + self.camera_distance
        factor = self.zoom / max(depth, 0.2)
        x = int(center_x + point[0] * factor)
        y = int(center_y - point[1] * factor)
        return x, y

    def render(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        frame_size = (width, height)

        ordered = sorted(self.shapes, key=lambda item: item.position[2], reverse=True)
        if self.preview_shape is not None:
            ordered.append(self.preview_shape)

        for shape in ordered:
            is_preview = self.preview_shape is not None and shape.shape_id == self.preview_shape.shape_id
            is_selected = self.selected_id == shape.shape_id and not is_preview

            vertices, edges = shape_geometry(shape.kind, shape.scale)
            transformed = []
            for vertex in vertices:
                rotated = rotate_y(vertex, shape.rotation_y)
                rotated = rotate_x(rotated, -0.35)
                transformed.append(rotated + np.array(shape.position, dtype=float))

            projected = [self.project(vertex, frame_size) for vertex in transformed]
            color = tuple(int(channel) for channel in shape.color)
            thickness = 3 if (is_selected or is_preview) else 2
            draw_color = color if not is_preview else tuple(min(channel + 35, 255) for channel in color)

            for start, end in edges:
                cv2.line(output, projected[start], projected[end], draw_color, thickness, cv2.LINE_AA)

            center = self.project(np.array(shape.position, dtype=float), frame_size)
            cv2.circle(output, center, 8 if is_selected else 6, draw_color, -1, cv2.LINE_AA)
            label = PRIMITIVE_LABELS[shape.kind]
            suffix = " (preview)" if is_preview else ""
            cv2.putText(
                output,
                f"{label}{suffix}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2,
                cv2.LINE_AA,
            )

        return output


def nearest_shape(
    scene: Scene3D,
    cursor: Tuple[int, int],
    frame_size: Tuple[int, int],
    max_distance: float,
) -> Optional[int]:
    nearest_id = None
    nearest_distance = 999999.0
    for shape in scene.shapes:
        center = scene.project(np.array(shape.position, dtype=float), frame_size)
        distance = math.dist(center, cursor)
        if distance < nearest_distance and distance < max_distance:
            nearest_distance = distance
            nearest_id = shape.shape_id
    return nearest_id
