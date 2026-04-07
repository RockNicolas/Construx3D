from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2

from .scene import GestureLatch, Scene3D, clamp, nearest_shape
from .settings import EXPORT_DIR, LATEST_JSON, PRIMITIVE_LABELS, WINDOW_NAME, ensure_runtime_dirs, get_display_work_area, latest_scene_input_path, load_settings
from .tracking import HandTracker, select_build_and_erase_hands
from .ui import draw_hold_indicator, draw_panel, export_scene


SHAPE_SELECTOR_ORDER = ["wall", "column", "slab", "stair", "roof"]


def pick_shape_from_cursor(cursor: tuple[int, int], frame_size: tuple[int, int]) -> Optional[str]:
    width, _ = frame_size
    x, y = cursor
    if y < 24 or y > 92:
        return None

    selector_left = max(width - 720, 280)
    selector_width = min(660, width - selector_left - 24)
    if x < selector_left or x > selector_left + selector_width:
        return None

    slot_width = selector_width / len(SHAPE_SELECTOR_ORDER)
    index = min(int((x - selector_left) / slot_width), len(SHAPE_SELECTOR_ORDER) - 1)
    return SHAPE_SELECTOR_ORDER[index]


def is_rotation_pose(build_hand, erase_hand) -> bool:
    if build_hand is None or erase_hand is None:
        return False
    if build_hand.pinch or erase_hand.pinch or erase_hand.is_erase_pose:
        return False
    return build_hand.finger_count >= 4 and erase_hand.finger_count >= 4


def main() -> None:
    settings = load_settings()
    ensure_runtime_dirs()

    scene = Scene3D(grid_size=settings.snap.grid_size, snap_enabled=settings.snap.enabled)
    latest_input = latest_scene_input_path()
    if latest_input.exists():
        scene.import_json(latest_input)

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

    zoom_anchor: Optional[Tuple[float, float]] = None
    rotation_anchor: Optional[Tuple[int, float, float]] = None
    pinch_was_active = False
    create_pose_was_active = False
    select_all_was_active = False
    hold_mode: Optional[str] = None
    active_shape_kind = "wall"
    last_erased_shape_id: Optional[int] = None
    status_text = "A mao rosa monta com pecas de construcao; a azul apaga ao passar por cima." 
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

            rotation_active = is_rotation_pose(build_hand, erase_hand) and scene.get_active_shape() is not None
            zoom_active = (
                build_hand is not None
                and erase_hand is not None
                and build_hand.pinch
                and erase_hand.pinch
                and scene.held_shape_id is None
            )

            if zoom_active:
                hand_distance = math.dist(build_hand.center, erase_hand.center)
                if zoom_anchor is None:
                    zoom_anchor = (hand_distance, scene.zoom)
                anchor_distance, anchor_zoom = zoom_anchor
                ratio = hand_distance / max(anchor_distance, 1.0)
                scene.zoom = clamp(anchor_zoom * ratio, settings.zoom.min_zoom, settings.zoom.max_zoom)
                status_text = f"Zoom ajustado para {int(scene.zoom)}"
            else:
                zoom_anchor = None

            if rotation_active:
                active_shape = scene.get_active_shape()
                if active_shape is not None:
                    hand_angle = math.atan2(
                        erase_hand.center[1] - build_hand.center[1],
                        erase_hand.center[0] - build_hand.center[0],
                    )
                    if rotation_anchor is None or rotation_anchor[0] != active_shape.shape_id:
                        scene.snapshot()
                        rotation_anchor = (active_shape.shape_id, hand_angle, active_shape.rotation_y)
                    shape_id, anchor_angle, anchor_rotation = rotation_anchor
                    scene.set_shape_rotation(shape_id, anchor_rotation + (hand_angle - anchor_angle))
                    status_text = "Rotacionando forma em 360 graus."
            else:
                rotation_anchor = None

            if build_hand is not None:
                selector_choice = pick_shape_from_cursor(build_hand.cursor, (frame_w, frame_h))
                if selector_choice is not None and selector_choice != active_shape_kind and not build_hand.pinch:
                    active_shape_kind = selector_choice
                    status_text = f"Peca selecionada: {PRIMITIVE_LABELS[active_shape_kind]}."

                cursor_world = scene.world_from_screen(build_hand.cursor, (frame_w, frame_h), build_hand.cursor_depth)
                undo_active = build_hand.gesture_matches([True, False, False, False, True])
                create_active = build_hand.is_create_pose and build_hover_id is None and not zoom_active and not rotation_active
                duplicate_active = build_hand.pinch and build_hover_id is not None and not zoom_active
                select_all_active = build_hand.is_select_all_pose and not zoom_active and not rotation_active and scene.held_shape_id is None

                if select_all_active and not select_all_was_active and scene.select_all():
                    status_text = f"{len(scene.shapes)} formas selecionadas."

                if create_active and not create_pose_was_active and scene.held_shape_id is None:
                    action = scene.begin_hold(cursor_world, active_shape_kind, None)
                    hold_mode = "create"
                    if action == "create":
                        status_text = f"{PRIMITIVE_LABELS[active_shape_kind]} criado com o gesto da mao rosa."
                elif hold_mode == "create" and scene.held_shape_id is not None:
                    if create_active:
                        if scene.update_held(cursor_world):
                            status_text = "Posicionando forma com o gesto da mao rosa."
                    else:
                        if scene.release_held():
                            status_text = "Forma criada e fixada na grade."
                        hold_mode = None

                if duplicate_active and not pinch_was_active and scene.held_shape_id is None:
                    action = scene.begin_hold(cursor_world, active_shape_kind, build_hover_id)
                    hold_mode = "pinch"
                    if action == "duplicate":
                        status_text = "Copia criada e presa ao dedo."
                elif hold_mode == "pinch" and scene.held_shape_id is not None:
                    if build_hand.pinch and not zoom_active:
                        if scene.update_held(cursor_world):
                            status_text = "Movendo copia em tempo real."
                    elif pinch_was_active and not zoom_active:
                        if scene.release_held():
                            status_text = "Copia fixada na ultima posicao."
                        hold_mode = None

                if latches.update(
                    "undo",
                    undo_active,
                    settings.gestures.undo.hold_time,
                    settings.gestures.undo.cooldown,
                    now,
                ):
                    if scene.undo():
                        status_text = "Ultima acao desfeita."

                cv2.circle(frame, build_hand.cursor, 10, (255, 225, 245), 2, cv2.LINE_AA)
                cv2.circle(frame, build_hand.cursor, 4, (255, 170, 235), -1, cv2.LINE_AA)
                pinch_was_active = build_hand.pinch
                create_pose_was_active = build_hand.is_create_pose
                select_all_was_active = select_all_active
            else:
                if pinch_was_active:
                    scene.release_held()
                if create_pose_was_active:
                    scene.release_held()
                pinch_was_active = False
                create_pose_was_active = False
                select_all_was_active = False
                hold_mode = None

            if erase_hand is not None:
                delete_active = erase_hand.is_erase_pose and erase_hover_id is not None
                if delete_active and erase_hover_id != last_erased_shape_id:
                    scene.hover_id = erase_hover_id
                    if scene.delete_focused():
                        last_erased_shape_id = erase_hover_id
                        status_text = "Forma apagada ao passar com a mao azul."
                elif not delete_active:
                    last_erased_shape_id = None

                cv2.circle(frame, erase_hand.cursor, 10, (255, 220, 190), 2, cv2.LINE_AA)
                cv2.circle(frame, erase_hand.cursor, 4, (255, 145, 95), -1, cv2.LINE_AA)
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
