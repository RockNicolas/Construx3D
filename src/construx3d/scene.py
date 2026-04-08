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


BLOCK_KIND = "square_1"
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


def module_spec(kind: str, scale: float) -> Tuple[float, float, float, int, int]:
    if kind == "square_1":
        return (scale, scale, scale, 1, 1)
    if kind == "square_3":
        return (scale * 3.0, scale, scale * 0.3, 3, 1)
    if kind == "square_5":
        return (scale * 5.0, scale, scale * 0.3, 5, 1)
    if kind == "square_5x3":
        return (scale * 5.0, scale * 3.0, scale * 0.3, 5, 3)
    return (scale, scale, scale, 1, 1)


def shape_geometry(kind: str, scale: float) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, ...]]]:
    width, height, depth, _, _ = module_spec(kind, scale)
    return build_box(width, height, depth)


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
    def __init__(
        self,
        grid_size: float = 0.85,
        snap_enabled: bool = True,
        camera_distance: float = 7.5,
        zoom: float = 560.0,
    ) -> None:
        self.shapes: List[Shape3D] = []
        self.held_shape_id: Optional[int] = None
        self.hover_id: Optional[int] = None
        self.selected_id: Optional[int] = None
        self.select_all_active = False
        self.next_id = 1
        self.camera_distance = camera_distance
        self.zoom = zoom
        self.camera_yaw = 0.0
        self.camera_pitch = 0.35
        self.history: List[Dict[str, object]] = []
        self.grid_size = max(grid_size, 0.05)
        self.snap_enabled = snap_enabled

    def _to_view_space(self, point: np.ndarray) -> np.ndarray:
        rotated = rotate_y(point, -self.camera_yaw)
        return rotate_x(rotated, -self.camera_pitch)

    def _from_view_space(self, point: np.ndarray) -> np.ndarray:
        rotated = rotate_x(point, self.camera_pitch)
        return rotate_y(rotated, self.camera_yaw)

    def orbit_camera(self, delta_x: float, delta_y: float) -> None:
        self.camera_yaw = (self.camera_yaw + delta_x) % (math.pi * 2.0)
        self.camera_pitch = clamp(self.camera_pitch + delta_y, -1.25, 1.25)

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
            "cube": "square_1",
            "wall": "square_1",
            "column": "square_3",
            "slab": "square_5",
            "stair": "square_5x3",
            "roof": "square_5x3",
            "pyramid": "square_5x3",
            "prism": "square_5",
        }
        kind = legacy_map.get(kind, kind)
        return kind if kind in AVAILABLE_SHAPES else BLOCK_KIND

    def _make_shape(self, kind: str, position: List[float]) -> Shape3D:
        normalized_kind = self._normalize_shape_kind(kind)
        return Shape3D(
            shape_id=self.next_id,
            kind=normalized_kind,
            position=self.snap_position(position),
            scale=self.grid_size,
            rotation_y=0.0,
            color=PRIMITIVE_COLORS[normalized_kind],
        )

    def snap_value(self, value: float, step: float) -> float:
        return round(value / step) * step

    def snap_position(self, position: List[float], step: Optional[float] = None) -> List[float]:
        if not self.snap_enabled:
            return list(position)

        active_step = max(step or self.grid_size, 0.05)
        return [self.snap_value(axis, active_step) for axis in position]

    def find_shape_at(self, position: List[float], step: Optional[float] = None) -> Optional[Shape3D]:
        snapped = self.snap_position(position, step)
        for shape in self.shapes:
            if all(abs(shape.position[axis] - snapped[axis]) <= 1e-6 for axis in range(3)):
                return shape
        return None

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

    def place_shape(self, position: List[float], kind: str = BLOCK_KIND, record_history: bool = True) -> bool:
        existing = self.find_shape_at(position)
        if existing is not None:
            self.selected_id = existing.shape_id
            self.select_all_active = False
            self.hover_id = existing.shape_id
            self.held_shape_id = None
            return False

        if record_history:
            self.snapshot()

        shape = self._make_shape(kind, position)
        shape.position = self.snap_position(position, shape.scale)
        self.shapes.append(shape)
        self.selected_id = shape.shape_id
        self.select_all_active = False
        self.hover_id = shape.shape_id
        self.held_shape_id = None
        self.next_id += 1
        return True

    def begin_move(self, shape_id: int, position: List[float]) -> bool:
        shape = next((item for item in self.shapes if item.shape_id == shape_id), None)
        if shape is None:
            return False

        if self.held_shape_id != shape_id:
            self.snapshot()

        self.held_shape_id = shape_id
        self.selected_id = shape_id
        self.select_all_active = False
        self.hover_id = None
        return self.update_held(position)

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

    def clear_selection(self) -> None:
        self.selected_id = None
        self.hover_id = None
        self.select_all_active = False

    def center_scene(self) -> bool:
        if not self.shapes:
            return False

        center = [
            sum(shape.position[axis] for shape in self.shapes) / len(self.shapes)
            for axis in range(3)
        ]
        for shape in self.shapes:
            shape.position = self.snap_position(
                [shape.position[axis] - center[axis] for axis in range(3)],
                shape.scale,
            )
        return True

    def begin_group_transform(self) -> bool:
        if not self.shapes:
            self.clear_selection()
            return False
        self.snapshot()
        self.select_all_active = True
        self.selected_id = self.shapes[-1].shape_id
        self.hover_id = None
        self.held_shape_id = None
        return True

    def apply_group_transform(
        self,
        base_positions: Dict[int, List[float]],
        base_rotations: Dict[int, float],
        group_center: List[float],
        translation: List[float],
        rotation_delta: float,
    ) -> bool:
        if not self.shapes:
            return False

        center = np.array(group_center, dtype=float)
        offset = np.array(translation, dtype=float)
        applied = False
        for shape in self.shapes:
            original_position = base_positions.get(shape.shape_id)
            if original_position is None:
                continue

            relative = np.array(original_position, dtype=float) - center
            rotated_relative = rotate_y(relative, rotation_delta)
            target_position = center + offset + rotated_relative
            shape.position = self.snap_position(target_position.tolist(), shape.scale)
            shape.rotation_y = (base_rotations.get(shape.shape_id, shape.rotation_y) + rotation_delta) % (math.pi * 2.0)
            applied = True

        return applied

    def select_shape(self, shape_id: int) -> bool:
        if any(shape.shape_id == shape_id for shape in self.shapes):
            self.selected_id = shape_id
            self.select_all_active = False
            self.hover_id = shape_id
            return True
        return False

    def get_shape(self, shape_id: Optional[int]) -> Optional[Shape3D]:
        if shape_id is None:
            return None
        for shape in self.shapes:
            if shape.shape_id == shape_id:
                return shape
        return None

    def get_selected(self) -> Optional[Shape3D]:
        if self.selected_id is None:
            return None
        shape = self.get_shape(self.selected_id)
        if shape is not None:
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
            "camera_yaw": self.camera_yaw,
            "camera_pitch": self.camera_pitch,
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
        self.camera_yaw = float(data.get("camera_yaw", self.camera_yaw))
        self.camera_pitch = float(data.get("camera_pitch", self.camera_pitch))
        self.center_scene()
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
        world = self._from_view_space(np.array([x, y, z], dtype=float))
        return world.tolist()

    def project(self, point: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = frame_size
        center_x = width / 2.0
        center_y = height / 2.0
        view_point = self._to_view_space(point)
        depth = view_point[2] + self.camera_distance
        factor = self.zoom / max(depth, 0.2)
        x = int(center_x + view_point[0] * factor)
        y = int(center_y - view_point[1] * factor)
        return x, y

    def render(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        frame_size = (width, height)

        ordered = sorted(
            self.shapes,
            key=lambda item: self._to_view_space(np.array(item.position, dtype=float))[2],
            reverse=True,
        )

        for shape in ordered:
            is_hovered = self.hover_id == shape.shape_id
            is_held = self.held_shape_id == shape.shape_id
            is_selected = self.select_all_active or self.selected_id == shape.shape_id

            vertices, edges, faces = shape_geometry(shape.kind, shape.scale)
            module_width, module_height, module_depth, module_cols, module_rows = module_spec(shape.kind, shape.scale)
            transformed = []
            for vertex in vertices:
                rotated = rotate_y(vertex, shape.rotation_y)
                transformed.append(rotated + np.array(shape.position, dtype=float))
            transformed_view = [self._to_view_space(vertex) for vertex in transformed]

            projected = [self.project(vertex, frame_size) for vertex in transformed]
            color = tuple(int(channel) for channel in shape.color)
            glow = 0.22 if is_hovered else 0.0
            glow += 0.28 if is_held else 0.12 if is_selected else 0.0
            outline_color = mix_color(color, (255, 240, 255), glow)
            accent_color = mix_color(color, (255, 255, 255), min(glow + 0.18, 0.45))
            thickness = 4 if is_held else 3 if (is_selected or is_hovered) else 2

            base_center = np.array(shape.position, dtype=float) + np.array([0.0, -shape.scale * 0.55, 0.0], dtype=float)
            shadow_center = self.project(base_center, frame_size)
            base_view_depth = self._to_view_space(base_center)[2]
            shadow_radius_x = max(int(shape.scale * self.zoom / max(base_view_depth + self.camera_distance + 0.8, 0.4) * 0.44), 18)
            shadow_radius_y = max(int(shadow_radius_x * 0.34), 10)
            shadow_layer = output.copy()
            cv2.ellipse(shadow_layer, shadow_center, (shadow_radius_x, shadow_radius_y), 0, 0, 360, (30, 20, 35), -1, cv2.LINE_AA)
            cv2.addWeighted(shadow_layer, 0.18 if is_held else 0.12, output, 0.88 if is_held else 0.92, 0.0, output)

            face_layer = output.copy()
            highlight_layer = output.copy()
            alpha = 0.05
            if is_hovered:
                alpha = 0.1
            if is_selected:
                alpha = max(alpha, 0.08)
            if is_held:
                alpha = 0.14

            sorted_faces = sorted(
                ((sum(transformed_view[index][2] for index in face) / len(face), face) for face in faces),
                reverse=True,
            )

            for depth_rank, (_, face) in enumerate(sorted_faces):
                polygon = np.array([projected[index] for index in face], dtype=np.int32)
                tone = 0.88 + depth_rank * 0.03
                face_color = scale_color(color, tone)
                if depth_rank == len(sorted_faces) - 1:
                    face_color = mix_color(face_color, (255, 255, 255), 0.1)
                elif depth_rank >= max(len(sorted_faces) - 3, 0):
                    face_color = scale_color(face_color, 0.96)
                cv2.fillPoly(face_layer, [polygon], face_color, lineType=cv2.LINE_AA)
                if depth_rank == len(sorted_faces) - 1:
                    cv2.fillPoly(highlight_layer, [polygon], accent_color, lineType=cv2.LINE_AA)

            cv2.addWeighted(face_layer, alpha, output, 1.0 - alpha, 0.0, output)
            cv2.addWeighted(highlight_layer, 0.09 if not is_held else 0.14, output, 0.91 if not is_held else 0.86, 0.0, output)

            for start, end in edges:
                edge_mid_depth = (transformed_view[start][2] + transformed_view[end][2]) / 2.0
                shape_mid_depth = self._to_view_space(np.array(shape.position, dtype=float))[2]
                is_front_edge = edge_mid_depth > shape_mid_depth
                edge_color = accent_color if is_front_edge else outline_color
                edge_thickness = thickness if is_front_edge else max(thickness - 1, 1)
                cv2.line(output, projected[start], projected[end], edge_color, edge_thickness, cv2.LINE_AA)

            grid_line_color = mix_color(outline_color, (255, 255, 255), 0.18)
            plane_depths = (-module_depth / 2.0, module_depth / 2.0)
            for plane_depth in plane_depths:
                for col_index in range(1, module_cols):
                    local_x = -module_width / 2.0 + (module_width * col_index / module_cols)
                    start_local = np.array([local_x, -module_height / 2.0, plane_depth], dtype=float)
                    end_local = np.array([local_x, module_height / 2.0, plane_depth], dtype=float)
                    start_world = rotate_y(start_local, shape.rotation_y) + np.array(shape.position, dtype=float)
                    end_world = rotate_y(end_local, shape.rotation_y) + np.array(shape.position, dtype=float)
                    cv2.line(output, self.project(start_world, frame_size), self.project(end_world, frame_size), grid_line_color, max(thickness - 1, 1), cv2.LINE_AA)

                for row_index in range(1, module_rows):
                    local_y = -module_height / 2.0 + (module_height * row_index / module_rows)
                    start_local = np.array([-module_width / 2.0, local_y, plane_depth], dtype=float)
                    end_local = np.array([module_width / 2.0, local_y, plane_depth], dtype=float)
                    start_world = rotate_y(start_local, shape.rotation_y) + np.array(shape.position, dtype=float)
                    end_world = rotate_y(end_local, shape.rotation_y) + np.array(shape.position, dtype=float)
                    cv2.line(output, self.project(start_world, frame_size), self.project(end_world, frame_size), grid_line_color, max(thickness - 1, 1), cv2.LINE_AA)

            center = self.project(np.array(shape.position, dtype=float), frame_size)
            cv2.circle(output, center, 8 if is_held else 6 if (is_selected or is_hovered) else 5, accent_color, -1, cv2.LINE_AA)
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
