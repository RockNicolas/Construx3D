from __future__ import annotations

import time

import cv2
import numpy as np

from .scene import Scene3D
from .settings import EXPORT_DIR, LATEST_JSON, PRIMITIVE_LABELS, SETTINGS_DISPLAY_PATH


def draw_panel(frame: np.ndarray, scene: Scene3D, active_shape_kind: str, status_text: str) -> None:
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (18, 18), (width - 18, 198), (20, 24, 32), -1)
    cv2.addWeighted(panel, 0.42, frame, 0.58, 0, frame)

    selected = scene.get_selected()
    selected_label = "Todas" if scene.select_all_active else PRIMITIVE_LABELS[selected.kind] if selected else "Nenhuma"
    held_label = PRIMITIVE_LABELS[scene.get_held().kind] if scene.get_held() else "Nenhum"
    lines = [
        f"Bloco ativo: {PRIMITIVE_LABELS[active_shape_kind]}",
        f"Selecionada: {selected_label}",
        f"Segurando: {held_label}",
        f"Blocos criados: {len(scene.shapes)} | Zoom: {int(scene.zoom)}",
        "Bloco fixo: 5x1 horizontal, igual ao da referencia",
        "Mao rosa: gesto de clicar com o indicador cria e fixa na hora",
        "Mao rosa: pinça sobre bloco duplica e arrasta a copia",
        "Mao rosa fechada seleciona todos os blocos criados",
        "Mao azul: indicador+medio apagam ao passar por cima | mao rosa: polegar+minimo desfaz",
        "Duas maos abertas rotacionam o bloco | duas pinças controlam zoom",
        f"Calibracao em: {SETTINGS_DISPLAY_PATH}",
        "Teclas: U desfaz, J exporta JSON, P exporta PNG, L importa ultimo JSON, ESC sai",
    ]

    cv2.putText(frame, "Construx3D", (34, 48), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        y = 76 + index * 18
        cv2.putText(frame, line, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 240), 1, cv2.LINE_AA)

    cv2.putText(frame, status_text, (34, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 255, 170), 2, cv2.LINE_AA)


def draw_hold_indicator(frame: np.ndarray, label: str, progress: float, row: int) -> None:
    if progress <= 0.0:
        return

    x = 24
    y = 214 + row * 24
    width = 220
    cv2.rectangle(frame, (x, y), (x + width, y + 16), (35, 40, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * progress), y + 16), (95, 220, 160), -1)
    cv2.putText(frame, label, (x + 6, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 18, 24), 1, cv2.LINE_AA)


def export_scene(scene: Scene3D, rendered_frame: np.ndarray) -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = EXPORT_DIR / f"scene_{timestamp}.json"
    png_path = EXPORT_DIR / f"scene_{timestamp}.png"
    scene.export_json(json_path)
    scene.export_json(LATEST_JSON)
    cv2.imwrite(str(png_path), rendered_frame)
    return f"Exportado: {json_path.name} e {png_path.name}"
