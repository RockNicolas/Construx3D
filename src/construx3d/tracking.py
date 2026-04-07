from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp_python = None
    mp_vision = None

from .settings import TrackingSettings, ensure_hand_landmarker_model


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandState:
    label: str
    landmarks_px: Dict[int, Tuple[int, int]]
    normalized: Dict[int, Tuple[float, float]]
    fingers: List[bool]
    pinch: bool
    pinch_distance: float
    cursor: Tuple[int, int]
    center: Tuple[int, int]

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)

    def gesture_matches(self, pattern: List[bool]) -> bool:
        return self.fingers == pattern


class HandTracker:
    def __init__(self, settings: TrackingSettings) -> None:
        self.settings = settings
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"

        if self.backend == "solutions":
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                model_complexity=self.settings.model_complexity,
                max_num_hands=self.settings.max_num_hands,
                min_detection_confidence=self.settings.min_detection_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
            )
            self.drawer = mp.solutions.drawing_utils
            self.landmarker = None
        else:
            if mp_python is None or mp_vision is None:
                raise RuntimeError(
                    "O pacote mediapipe instalado nao oferece nem a API solutions nem a API tasks necessaria."
                )

            model_path = ensure_hand_landmarker_model()
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=self.settings.max_num_hands,
                min_hand_detection_confidence=self.settings.min_detection_confidence,
                min_hand_presence_confidence=self.settings.min_detection_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self.hands = None
            self.mp_hands = None
            self.drawer = None

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.landmarker is not None:
            self.landmarker.close()

    def detect(self, frame: np.ndarray) -> List[HandState]:
        if self.backend == "solutions":
            return self._detect_with_solutions(frame)
        return self._detect_with_tasks(frame)

    def _detect_with_solutions(self, frame: np.ndarray) -> List[HandState]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detected: List[HandState] = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detected

        frame_h, frame_w = frame.shape[:2]
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            landmarks_px: Dict[int, Tuple[int, int]] = {}
            normalized: Dict[int, Tuple[float, float]] = {}
            for index, landmark in enumerate(hand_landmarks.landmark):
                px = int(landmark.x * frame_w)
                py = int(landmark.y * frame_h)
                landmarks_px[index] = (px, py)
                normalized[index] = (landmark.x, landmark.y)

            thumb_tip = landmarks_px[4]
            index_tip = landmarks_px[8]
            pinch_distance = math.dist(thumb_tip, index_tip)
            palm_center = landmarks_px[9]
            fingers = self._finger_state(landmarks_px, label)
            detected.append(
                HandState(
                    label=label,
                    landmarks_px=landmarks_px,
                    normalized=normalized,
                    fingers=fingers,
                    pinch=pinch_distance < self.settings.pinch_distance_threshold_px,
                    pinch_distance=pinch_distance,
                    cursor=index_tip,
                    center=palm_center,
                )
            )
            self.drawer.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(color=(120, 255, 190), thickness=2, circle_radius=3),
                self.drawer.DrawingSpec(color=(70, 120, 255), thickness=2),
            )

        return detected

    def _detect_with_tasks(self, frame: np.ndarray) -> List[HandState]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected: List[HandState] = []

        if not result.hand_landmarks or not result.handedness:
            return detected

        frame_h, frame_w = frame.shape[:2]
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = self._extract_handedness_label(handedness)
            landmarks_px: Dict[int, Tuple[int, int]] = {}
            normalized: Dict[int, Tuple[float, float]] = {}

            for index, landmark in enumerate(hand_landmarks):
                px = int(landmark.x * frame_w)
                py = int(landmark.y * frame_h)
                landmarks_px[index] = (px, py)
                normalized[index] = (landmark.x, landmark.y)

            thumb_tip = landmarks_px[4]
            index_tip = landmarks_px[8]
            pinch_distance = math.dist(thumb_tip, index_tip)
            palm_center = landmarks_px[9]
            fingers = self._finger_state(landmarks_px, label)
            detected.append(
                HandState(
                    label=label,
                    landmarks_px=landmarks_px,
                    normalized=normalized,
                    fingers=fingers,
                    pinch=pinch_distance < self.settings.pinch_distance_threshold_px,
                    pinch_distance=pinch_distance,
                    cursor=index_tip,
                    center=palm_center,
                )
            )
            self._draw_task_landmarks(frame, landmarks_px)

        return detected

    def _draw_task_landmarks(self, frame: np.ndarray, landmarks_px: Dict[int, Tuple[int, int]]) -> None:
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, landmarks_px[start], landmarks_px[end], (70, 120, 255), 2, cv2.LINE_AA)
        for point in landmarks_px.values():
            cv2.circle(frame, point, 3, (120, 255, 190), -1, cv2.LINE_AA)

    def _extract_handedness_label(self, handedness) -> str:
        if not handedness:
            return "Right"
        first = handedness[0]
        return getattr(first, "category_name", None) or getattr(first, "display_name", None) or "Right"

    def _finger_state(self, landmarks_px: Dict[int, Tuple[int, int]], handedness: str) -> List[bool]:
        fingers = [False] * 5
        thumb_tip_x = landmarks_px[4][0]
        thumb_joint_x = landmarks_px[3][0]
        if handedness == "Right":
            fingers[0] = thumb_tip_x > thumb_joint_x
        else:
            fingers[0] = thumb_tip_x < thumb_joint_x

        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        for offset, (tip_id, pip_id) in enumerate(zip(tip_ids, pip_ids), start=1):
            fingers[offset] = landmarks_px[tip_id][1] < landmarks_px[pip_id][1]
        return fingers


def select_support_and_action_hands(hands: List[HandState]) -> Tuple[Optional[HandState], Optional[HandState]]:
    support = next((hand for hand in hands if hand.label == "Left"), None)
    action = next((hand for hand in hands if hand.label == "Right"), None)

    if len(hands) == 1:
        single = hands[0]
        if single.label == "Left":
            return support, None
        return None, single

    return support, action
