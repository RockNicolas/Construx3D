from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import cv2

from .scene import GestureLatch, Scene3D, clamp, nearest_shape
from .settings import EXPORT_DIR, LATEST_JSON, PRIMITIVE_LABELS, WINDOW_NAME, ensure_runtime_dirs, get_display_work_area, latest_scene_input_path, load_settings
from .tracking import HandTracker, select_build_and_erase_hands
from .ui import draw_hold_indicator, draw_panel, export_scene


FIXED_SHAPE_KIND = "square_1"


def cursor_colors(cursor_x: int, frame_width: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if cursor_x < frame_width // 2:
        return (255, 225, 245), (255, 170, 235)
    return (255, 225, 205), (255, 170, 120)

def main() -> None:
    settings = load_settings()
    ensure_runtime_dirs()

    scene = Scene3D(
        grid_size=settings.snap.grid_size,
        snap_enabled=settings.snap.enabled,
        camera_distance=settings.zoom.camera_distance,
        zoom=settings.zoom.default_zoom,
    )
    latest_input = latest_scene_input_path()
    if latest_input.exists():
        scene.import_json(latest_input)
    else:
        scene.place_shape([0.0, 0.0, 0.0], FIXED_SHAPE_KIND)

    tracker = HandTracker(settings.tracking)
    latches = GestureLatch()
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera.width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera.height)

    if not webcam.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    work_x, work_y, work_width, work_height = get_display_work_area()
    initial_width = min(settings.camera.width, max(work_width - 80, 640))
    initial_height = min(settings.camera.height, max(work_height - 80, 480))
    initial_x = work_x + max((work_width - initial_width) // 2, 0)
    initial_y = work_y + max((work_height - initial_height) // 2, 0)
    cv2.resizeWindow(WINDOW_NAME, initial_width, initial_height)
    cv2.moveWindow(WINDOW_NAME, initial_x, initial_y)

    create_pose_was_active = False
    freeze_pose_was_active = False
    manipulation_anchor: Optional[Dict[str, Any]] = None
    active_shape_kind = FIXED_SHAPE_KIND
    last_erased_shape_id: Optional[int] = None
    status_text = "Mao rosa aberta move e gira todos os blocos; mao azul apaga."
    previous_time = time.time()

    try:
        while True:
            ok, frame = webcam.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)
            build_hand, erase_hand = select_build_and_erase_hands(hands)
            now = time.time()
            frame_h, frame_w = frame.shape[:2]

            build_hover_id = None
            erase_hover_id = None

            if build_hand is not None:
                build_hover_id = None if scene.held_shape_id is not None else nearest_shape(
                    scene,
                    build_hand.cursor,
                    (frame_w, frame_h),
                    settings.selection.max_cursor_distance_px,
                )

            if erase_hand is not None:
                erase_hover_id = None if scene.held_shape_id is not None else nearest_shape(
                    scene,
                    erase_hand.cursor,
                    (frame_w, frame_h),
                    settings.selection.max_cursor_distance_px,
                )

            scene.hover_id = erase_hover_id if erase_hover_id is not None else build_hover_id

            undo_active = False
            freeze_active = False
            if build_hand is not None:
                undo_active = build_hand.gesture_matches([True, False, False, False, True])
                freeze_active = build_hand.is_fist and not undo_active

            if build_hand is not None:
                cursor_world = scene.world_from_screen(build_hand.center, (frame_w, frame_h), build_hand.cursor_depth)
                create_active = False
                select_active = build_hand.is_create_pose and build_hover_id is not None and not freeze_active
                manipulation_active = build_hand.is_open_palm and bool(scene.shapes) and not undo_active and not freeze_active

                if freeze_active:
                    manipulation_anchor = None
                    if not freeze_pose_was_active:
                        scene.clear_selection()
                        status_text = "Tudo parado. Nenhum bloco ativo."

                if select_active and not create_pose_was_active and scene.held_shape_id is None:
                    if scene.select_shape(build_hover_id):
                        status_text = f"{PRIMITIVE_LABELS[active_shape_kind]} selecionada."

                if manipulation_active:
                    if manipulation_anchor is None:
                        if scene.begin_group_transform():
                            manipulation_anchor = {
                                "hand_world": cursor_world,
                                "hand_x": build_hand.center[0],
                                "base_positions": {shape.shape_id: list(shape.position) for shape in scene.shapes},
                                "base_rotations": {shape.shape_id: shape.rotation_y for shape in scene.shapes},
                                "group_center": [
                                    sum(shape.position[axis] for shape in scene.shapes) / len(scene.shapes)
                                    for axis in range(3)
                                ],
                            }
                    if manipulation_anchor is not None:
                        translation = [
                            cursor_world[axis] - manipulation_anchor["hand_world"][axis]
                            for axis in range(3)
                        ]
                        rotation_delta = (build_hand.center[0] - manipulation_anchor["hand_x"]) * (math.pi / max(frame_w, 1)) * 2.0
                        if scene.apply_group_transform(
                            manipulation_anchor["base_positions"],
                            manipulation_anchor["base_rotations"],
                            manipulation_anchor["group_center"],
                            translation,
                            rotation_delta,
                        ):
                            status_text = "Movendo e girando todos os blocos com a mao rosa aberta."
                else:
                    manipulation_anchor = None

                if latches.update(
                    "undo",
                    undo_active,
                    settings.gestures.undo.hold_time,
                    settings.gestures.undo.cooldown,
                    now,
                ):
                    if scene.undo():
                        status_text = "Ultima acao desfeita."

                outer_color, inner_color = cursor_colors(build_hand.cursor[0], frame_w)
                cv2.circle(frame, build_hand.cursor, 10, outer_color, 2, cv2.LINE_AA)
                cv2.circle(frame, build_hand.cursor, 4, inner_color, -1, cv2.LINE_AA)
                create_pose_was_active = build_hand.is_create_pose
                freeze_pose_was_active = freeze_active
            else:
                manipulation_anchor = None
                create_pose_was_active = False
                freeze_pose_was_active = False

            if erase_hand is not None:
                delete_active = erase_hand.is_erase_pose and erase_hover_id is not None
                if delete_active and erase_hover_id != last_erased_shape_id:
                    scene.hover_id = erase_hover_id
                    if scene.delete_focused():
                        last_erased_shape_id = erase_hover_id
                        status_text = "Forma apagada ao passar com a mao azul."
                elif not delete_active:
                    last_erased_shape_id = None

                outer_color, inner_color = cursor_colors(erase_hand.cursor[0], frame_w)
                cv2.circle(frame, erase_hand.cursor, 10, outer_color, 2, cv2.LINE_AA)
                cv2.circle(frame, erase_hand.cursor, 4, inner_color, -1, cv2.LINE_AA)
            elif build_hand is None:
                scene.hover_id = None
                last_erased_shape_id = None

            rendered = scene.render(frame)

            if erase_hand is not None:
                delete_progress = 1.0 if erase_hand.is_erase_pose else 0.0
                draw_hold_indicator(rendered, "Apagar", delete_progress, 0)

            if build_hand is not None:
                undo_progress = latches.progress(
                    "undo",
                    build_hand.gesture_matches([True, False, False, False, True]),
                    settings.gestures.undo.hold_time,
                    now,
                )
                draw_hold_indicator(rendered, "Desfazer", undo_progress, 1)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            cv2.putText(rendered, f"FPS: {int(fps)}", (frame_w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            draw_panel(rendered, scene, active_shape_kind, status_text)
            cv2.imshow(WINDOW_NAME, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("u"), ord("U")):
                if scene.undo():
                    status_text = "Ultima alteracao desfeita."
            if key in (ord("j"), ord("J")):
                scene.export_json(LATEST_JSON)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                export_path = EXPORT_DIR / f"scene_{timestamp}.json"
                scene.export_json(export_path)
                status_text = f"JSON exportado para {export_path.name}"
            if key in (ord("p"), ord("P")):
                status_text = export_scene(scene, rendered)
            if key in (ord("l"), ord("L")):
                if scene.import_json(latest_scene_input_path()):
                    status_text = "Ultimo JSON recarregado."
                else:
                    status_text = "Nenhum arquivo latest_scene.json encontrado."

    finally:
        webcam.release()
        tracker.close()
        cv2.destroyAllWindows()
