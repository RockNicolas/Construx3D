from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2

from .scene import GestureLatch, Scene3D, clamp, nearest_shape
from .settings import EXPORT_DIR, LATEST_JSON, PRIMITIVES, PRIMITIVE_LABELS, WINDOW_NAME, ensure_runtime_dirs, get_display_work_area, latest_scene_input_path, load_settings
from .tracking import HandTracker, select_support_and_action_hands
from .ui import draw_hold_indicator, draw_panel, export_scene


def main() -> None:
    settings = load_settings()
    ensure_runtime_dirs()

    scene = Scene3D()
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

    action_dragging = False
    drag_snapshot_taken = False
    zoom_anchor: Optional[Tuple[float, float]] = None
    active_shape_kind = "cube"
    status_text = "Mostre a mao direita para selecionar e a esquerda para definir o tipo."
    previous_time = time.time()

    try:
        while True:
            ok, frame = webcam.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)
            support_hand, action_hand = select_support_and_action_hands(hands)
            now = time.time()
            frame_h, frame_w = frame.shape[:2]

            if support_hand and not support_hand.pinch:
                active_shape_kind = PRIMITIVES.get(support_hand.finger_count, active_shape_kind)

            if support_hand and support_hand.pinch:
                if zoom_anchor is None:
                    zoom_anchor = (support_hand.pinch_distance, scene.zoom)
                anchor_distance, anchor_zoom = zoom_anchor
                ratio = support_hand.pinch_distance / max(anchor_distance, 1.0)
                scene.zoom = clamp(anchor_zoom * ratio, settings.zoom.min_zoom, settings.zoom.max_zoom)
                status_text = f"Zoom ajustado para {int(scene.zoom)}"
            else:
                zoom_anchor = None

            if action_hand is not None:
                nearest_id = nearest_shape(
                    scene,
                    action_hand.cursor,
                    (frame_w, frame_h),
                    settings.selection.max_cursor_distance_px,
                )
                if not action_dragging and scene.preview_shape is None and nearest_id is not None:
                    scene.selected_id = nearest_id

                cursor_world = scene.world_from_screen(action_hand.cursor, (frame_w, frame_h))

                create_active = action_hand.gesture_matches([False, True, True, False, False])
                duplicate_active = action_hand.gesture_matches([False, True, True, True, False])
                delete_active = action_hand.finger_count == 0
                undo_active = action_hand.gesture_matches([True, False, False, False, True])
                commit_active = action_hand.finger_count >= 4 and scene.preview_shape is not None

                if scene.preview_shape is not None:
                    scene.preview_shape.position = cursor_world
                    if latches.update(
                        "commit",
                        commit_active,
                        settings.gestures.commit.hold_time,
                        settings.gestures.commit.cooldown,
                        now,
                    ):
                        scene.commit_preview()
                        status_text = "Estrutura fixada na cena."
                    if latches.update(
                        "cancel_preview",
                        delete_active,
                        settings.gestures.cancel_preview.hold_time,
                        settings.gestures.cancel_preview.cooldown,
                        now,
                    ):
                        scene.cancel_preview()
                        status_text = "Preview cancelado."
                else:
                    if latches.update(
                        "create",
                        create_active,
                        settings.gestures.create.hold_time,
                        settings.gestures.create.cooldown,
                        now,
                    ):
                        scene.create_preview(active_shape_kind, cursor_world)
                        status_text = f"Preview de {PRIMITIVE_LABELS[active_shape_kind]} criado."

                    if latches.update(
                        "duplicate",
                        duplicate_active,
                        settings.gestures.duplicate.hold_time,
                        settings.gestures.duplicate.cooldown,
                        now,
                    ):
                        if scene.duplicate_selected_to_preview():
                            status_text = "Copia criada. Reposicione com pinch e abra a mao para fixar."

                    if latches.update(
                        "delete",
                        delete_active,
                        settings.gestures.delete.hold_time,
                        settings.gestures.delete.cooldown,
                        now,
                    ):
                        if scene.delete_selected():
                            status_text = "Estrutura deletada."

                    if action_hand.pinch and scene.get_selected() is not None:
                        selected_shape = scene.get_selected()
                        if selected_shape is not None:
                            if not drag_snapshot_taken:
                                scene.snapshot()
                                drag_snapshot_taken = True
                            selected_shape.position = cursor_world
                            action_dragging = True
                            status_text = "Reposicionando estrutura selecionada."
                    else:
                        action_dragging = False
                        drag_snapshot_taken = False

                if latches.update(
                    "undo",
                    undo_active,
                    settings.gestures.undo.hold_time,
                    settings.gestures.undo.cooldown,
                    now,
                ):
                    if scene.undo():
                        status_text = "Ultima acao desfeita."

                cv2.circle(frame, action_hand.cursor, 10, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, action_hand.cursor, 4, (90, 255, 160), -1, cv2.LINE_AA)

            rendered = scene.render(frame)

            if action_hand is not None:
                create_progress = latches.progress(
                    "create",
                    action_hand.gesture_matches([False, True, True, False, False]) and scene.preview_shape is None,
                    settings.gestures.create.hold_time,
                    now,
                )
                duplicate_progress = latches.progress(
                    "duplicate",
                    action_hand.gesture_matches([False, True, True, True, False]) and scene.preview_shape is None,
                    settings.gestures.duplicate.hold_time,
                    now,
                )
                delete_progress = latches.progress(
                    "delete",
                    action_hand.finger_count == 0,
                    settings.gestures.delete.hold_time,
                    now,
                )
                undo_progress = latches.progress(
                    "undo",
                    action_hand.gesture_matches([True, False, False, False, True]),
                    settings.gestures.undo.hold_time,
                    now,
                )
                commit_progress = latches.progress(
                    "commit",
                    action_hand.finger_count >= 4 and scene.preview_shape is not None,
                    settings.gestures.commit.hold_time,
                    now,
                )
                draw_hold_indicator(rendered, "Criar", create_progress, 0)
                draw_hold_indicator(rendered, "Duplicar", duplicate_progress, 1)
                draw_hold_indicator(rendered, "Deletar", delete_progress, 2)
                draw_hold_indicator(rendered, "Desfazer", undo_progress, 3)
                draw_hold_indicator(rendered, "Fixar", commit_progress, 4)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            cv2.putText(rendered, f"FPS: {int(fps)}", (frame_w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            draw_panel(rendered, scene, active_shape_kind, status_text)
            cv2.imshow(WINDOW_NAME, rendered)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
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
