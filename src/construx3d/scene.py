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


BLOCK_KIND = "wall"
AVAILABLE_SHAPES = tuple(PRIMITIVE_LABELS.keys())


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


def build_box(width: float, height: float, depth: float) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, ...]]]:
    half_w = width / 2.0
    half_h = height / 2.0
    half_d = depth / 2.0
    vertices = [
        np.array([x, y, z], dtype=float)
        for x in (-half_w, half_w)
        for y in (-half_h, half_h)
        for z in (-half_d, half_d)
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
    faces = [
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    ]
    return vertices, edges, faces


def shape_geometry(kind: str, scale: float) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, ...]]]:
    if kind == "wall":
        return build_box(scale * 1.9, scale * 1.2, scale * 0.28)

    if kind == "column":
        return build_box(scale * 0.6, scale * 1.9, scale * 0.6)

    if kind == "slab":
        return build_box(scale * 1.9, scale * 0.3, scale * 1.9)

    if kind == "roof":
        half_w = scale * 0.95
        half_h = scale * 0.48
        half_d = scale * 0.7
        vertices = [
            np.array([-half_w, -half_h, -half_d], dtype=float),
            np.array([half_w, -half_h, -half_d], dtype=float),
            np.array([0.0, half_h, -half_d], dtype=float),
            np.array([-half_w, -half_h, half_d], dtype=float),
            np.array([half_w, -half_h, half_d], dtype=float),
            np.array([0.0, half_h, half_d], dtype=float),
        ]
        edges = [
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (0, 3), (1, 4), (2, 5),
        ]
        faces = [
            (0, 1, 2),
            (3, 4, 5),
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (2, 0, 3, 5),
        ]
        return vertices, edges, faces

    if kind == "stair":
        half_w = scale * 0.9
        half_d = scale * 0.7
        low_y = -scale * 0.48
        mid_y = -scale * 0.08
        high_y = scale * 0.32
        vertices = [
            np.array([-half_w, low_y, -half_d], dtype=float),
            np.array([half_w, low_y, -half_d], dtype=float),
            np.array([half_w, mid_y, -half_d], dtype=float),
            np.array([0.0, mid_y, -half_d], dtype=float),
            np.array([0.0, high_y, -half_d], dtype=float),
            np.array([-half_w, high_y, -half_d], dtype=float),
            np.array([-half_w, low_y, half_d], dtype=float),
            np.array([half_w, low_y, half_d], dtype=float),
            np.array([half_w, mid_y, half_d], dtype=float),
            np.array([0.0, mid_y, half_d], dtype=float),
            np.array([0.0, high_y, half_d], dtype=float),
            np.array([-half_w, high_y, half_d], dtype=float),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
            (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
        ]
        faces = [
            (0, 1, 7, 6),
            (1, 2, 8, 7),
            (2, 3, 9, 8),
            (3, 4, 10, 9),
            (4, 5, 11, 10),
            (0, 5, 11, 6),
            (0, 1, 2, 3, 4, 5),
            (6, 7, 8, 9, 10, 11),
        ]
        return vertices, edges, faces

    return build_box(scale, scale, scale)


def mix_color(color: Tuple[int, int, int], target: Tuple[int, int, int], weight: float) -> Tuple[int, int, int]:
    return tuple(
        int(channel * (1.0 - weight) + target_channel * weight)
        for channel, target_channel in zip(color, target)
    )


def scale_color(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(int(clamp(channel * factor, 0, 255)) for channel in color)


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
    def __init__(self, grid_size: float = 1.25, snap_enabled: bool = True) -> None:
        self.shapes: List[Shape3D] = []
        self.held_shape_id: Optional[int] = None
        self.hover_id: Optional[int] = None
        self.selected_id: Optional[int] = None
        self.select_all_active = False
        self.next_id = 1
        self.camera_distance = 7.5
        self.zoom = 700.0
        self.history: List[Dict[str, object]] = []
        self.grid_size = max(grid_size, 0.05)
        self.snap_enabled = snap_enabled

    def snapshot(self) -> None:
        state = {
            "shapes": copy.deepcopy([asdict(shape) for shape in self.shapes]),
            "selected_id": self.selected_id,
            "select_all_active": self.select_all_active,
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
        self.select_all_active = bool(previous.get("select_all_active", False))
        self.next_id = previous["next_id"]
        self.zoom = float(previous["zoom"])
        self.held_shape_id = None
        self.hover_id = self.selected_id
        return True

    def _normalize_shape_kind(self, kind: str) -> str:
        legacy_map = {
            "cube": "wall",
            "pyramid": "roof",
            "prism": "stair",
        }
        kind = legacy_map.get(kind, kind)
        return kind if kind in AVAILABLE_SHAPES else BLOCK_KIND

    def _make_shape(self, kind: str, position: List[float]) -> Shape3D:
        normalized_kind = self._normalize_shape_kind(kind)
        return Shape3D(
            shape_id=self.next_id,
            kind=normalized_kind,
            position=self.snap_position(position),
            scale=1.25,
            rotation_y=0.45,
            color=PRIMITIVE_COLORS[normalized_kind],
        )

    def snap_value(self, value: float, step: float) -> float:
        return round(value / step) * step

    def snap_position(self, position: List[float], step: Optional[float] = None) -> List[float]:
        if not self.snap_enabled:
            return list(position)

        active_step = max(step or self.grid_size, 0.05)
        return [self.snap_value(axis, active_step) for axis in position]

    def begin_hold(self, position: List[float], kind: str = BLOCK_KIND, source_id: Optional[int] = None) -> str:
        source = next((shape for shape in self.shapes if shape.shape_id == source_id), None)

        self.snapshot()
        shape = self._make_shape(kind, position)
        action = "create"
        if source is not None:
            shape = copy.deepcopy(source)
            shape.shape_id = self.next_id
            shape.position = list(source.position)
            action = "duplicate"

        self.shapes.append(shape)
        self.held_shape_id = shape.shape_id
        self.selected_id = shape.shape_id
        self.select_all_active = False
        self.hover_id = None
        self.next_id += 1
        self.update_held(position)
        return action

    def update_held(self, position: List[float]) -> bool:
        held = self.get_held()
        if held is None:
            return False
        held.position = self.snap_position(position, held.scale)
        return True

    def release_held(self) -> bool:
        held = self.get_held()
        if held is None:
            return False
        held.position = self.snap_position(held.position, held.scale)
        self.selected_id = self.held_shape_id
        self.held_shape_id = None
        return True

    def select_all(self) -> bool:
        if not self.shapes:
            self.select_all_active = False
            self.selected_id = None
            return False
        self.select_all_active = True
        self.selected_id = self.shapes[-1].shape_id
        return True

    def clear_select_all(self) -> None:
        self.select_all_active = False

    def get_selected(self) -> Optional[Shape3D]:
        if self.selected_id is None:
            return None
        for shape in self.shapes:
            if shape.shape_id == self.selected_id:
                return shape
        self.selected_id = None
        return None

    def get_held(self) -> Optional[Shape3D]:
        if self.held_shape_id is None:
            return None
        for shape in self.shapes:
            if shape.shape_id == self.held_shape_id:
                return shape
        self.held_shape_id = None
        return None

    def get_active_shape(self) -> Optional[Shape3D]:
        held = self.get_held()
        if held is not None:
            return held
        return self.get_selected()

    def set_shape_rotation(self, shape_id: int, rotation_y: float) -> bool:
        for shape in self.shapes:
            if shape.shape_id == shape_id:
                shape.rotation_y = rotation_y % (math.pi * 2.0)
                return True
        return False

    def delete_focused(self) -> bool:
        target_id = self.hover_id if self.hover_id is not None else self.selected_id
        if target_id is None:
            return False

        remaining = [shape for shape in self.shapes if shape.shape_id != target_id]
        if len(remaining) == len(self.shapes):
            return False

        self.snapshot()
        self.shapes = remaining
        if self.held_shape_id == target_id:
            self.held_shape_id = None
        self.hover_id = None
        self.select_all_active = False
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
        normalized_shapes = []
        for shape_data in data.get("shapes", []):
            normalized = dict(shape_data)
            normalized["kind"] = self._normalize_shape_kind(str(normalized.get("kind", BLOCK_KIND)))
            normalized.setdefault("color", PRIMITIVE_COLORS[normalized["kind"]])
            normalized_shapes.append(Shape3D(**normalized))

        self.shapes = normalized_shapes
        self.next_id = int(data.get("next_id", len(self.shapes) + 1))
        self.zoom = float(data.get("zoom", self.zoom))
        self.camera_distance = float(data.get("camera_distance", self.camera_distance))
        self.selected_id = self.shapes[-1].shape_id if self.shapes else None
        self.select_all_active = False
        self.held_shape_id = None
        self.hover_id = self.selected_id
        self.history.clear()
        return True

    def world_from_screen(self, point: Tuple[int, int], frame_size: Tuple[int, int], depth: float = 0.0) -> List[float]:
        width, height = frame_size
        center_x = width / 2.0
        center_y = height / 2.0
        x = (point[0] - center_x) * self.camera_distance / self.zoom
        y = (center_y - point[1]) * self.camera_distance / self.zoom
        z = clamp(-depth * self.camera_distance * 10.0, -2.5, 2.5)
        return [x, y, z]

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

        for shape in ordered:
            is_hovered = self.hover_id == shape.shape_id
            is_held = self.held_shape_id == shape.shape_id
            is_selected = self.select_all_active or self.selected_id == shape.shape_id

            vertices, edges, faces = shape_geometry(shape.kind, shape.scale)
            transformed = []
            for vertex in vertices:
                rotated = rotate_y(vertex, shape.rotation_y)
                rotated = rotate_x(rotated, -0.35)
                transformed.append(rotated + np.array(shape.position, dtype=float))

            projected = [self.project(vertex, frame_size) for vertex in transformed]
            color = tuple(int(channel) for channel in shape.color)
            glow = 0.22 if is_hovered else 0.0
            glow += 0.28 if is_held else 0.12 if is_selected else 0.0
            outline_color = mix_color(color, (255, 240, 255), glow)
            accent_color = mix_color(color, (255, 255, 255), min(glow + 0.18, 0.45))
            thickness = 4 if is_held else 3 if (is_selected or is_hovered) else 2

            base_center = np.array(shape.position, dtype=float) + np.array([0.0, -shape.scale * 0.55, 0.0], dtype=float)
            shadow_center = self.project(base_center, frame_size)
            shadow_radius_x = max(int(shape.scale * self.zoom / max(shape.position[2] + self.camera_distance + 0.8, 0.4) * 0.44), 18)
            shadow_radius_y = max(int(shadow_radius_x * 0.34), 10)
            shadow_layer = output.copy()
            cv2.ellipse(shadow_layer, shadow_center, (shadow_radius_x, shadow_radius_y), 0, 0, 360, (30, 20, 35), -1, cv2.LINE_AA)
            cv2.addWeighted(shadow_layer, 0.18 if is_held else 0.12, output, 0.88 if is_held else 0.92, 0.0, output)

            face_layer = output.copy()
            highlight_layer = output.copy()
            alpha = 0.18
            if is_hovered:
                alpha = 0.26
            if is_selected:
                alpha = max(alpha, 0.24)
            if is_held:
                alpha = 0.34

            sorted_faces = sorted(
                ((sum(transformed[index][2] for index in face) / len(face), face) for face in faces),
                reverse=True,
            )

            for depth_rank, (_, face) in enumerate(sorted_faces):
                polygon = np.array([projected[index] for index in face], dtype=np.int32)
                tone = 0.76 + depth_rank * 0.06
                face_color = scale_color(color, tone)
                if depth_rank == len(sorted_faces) - 1:
                    face_color = mix_color(face_color, (255, 255, 255), 0.18)
                elif depth_rank >= max(len(sorted_faces) - 3, 0):
                    face_color = scale_color(face_color, 0.92)
                cv2.fillPoly(face_layer, [polygon], face_color, lineType=cv2.LINE_AA)
                if depth_rank == len(sorted_faces) - 1:
                    cv2.fillPoly(highlight_layer, [polygon], accent_color, lineType=cv2.LINE_AA)

            cv2.addWeighted(face_layer, alpha, output, 1.0 - alpha, 0.0, output)
            cv2.addWeighted(highlight_layer, 0.09 if not is_held else 0.14, output, 0.91 if not is_held else 0.86, 0.0, output)

            for start, end in edges:
                edge_mid_depth = (transformed[start][2] + transformed[end][2]) / 2.0
                is_front_edge = edge_mid_depth > shape.position[2]
                edge_color = accent_color if is_front_edge else outline_color
                edge_thickness = thickness if is_front_edge else max(thickness - 1, 1)
                cv2.line(output, projected[start], projected[end], edge_color, edge_thickness, cv2.LINE_AA)

            center = self.project(np.array(shape.position, dtype=float), frame_size)
            cv2.circle(output, center, 10 if is_held else 8 if (is_selected or is_hovered) else 6, accent_color, -1, cv2.LINE_AA)
            if is_hovered or is_held:
                halo_layer = output.copy()
                cv2.circle(halo_layer, center, 18 if is_held else 14, outline_color, 2, cv2.LINE_AA)
                cv2.addWeighted(halo_layer, 0.3, output, 0.7, 0.0, output)

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
