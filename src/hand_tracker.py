"""
Módulo de detección y extracción de landmarks de manos utilizando MediaPipe.
Compatible de forma nativa tanto con MediaPipe Tasks API (Python 3.12/3.13+)
como con la API clásica mp.solutions.hands (Python 3.10/3.11).
"""

import os
import urllib.request
from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "hand_landmarker.task",
)


def ensure_model_file(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Asegura que el archivo binario del modelo preentrenado esté disponible localmente."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[HandTracker] Descargando modelo preentrenado de MediaPipe en {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[HandTracker] Descarga del modelo completada exitosamente.")
    return model_path


class HandTracker:
    """
    Encapsula la detección de manos en tiempo real y extrae las coordenadas
    de las puntas de los dedos ([4, 8, 12, 16, 20]).
    """

    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Pulgar, Índice, Medio, Anular, Meñique

    # Pares de conexiones de articulaciones para dibujo de depuración
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
        (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
        (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
        (9, 13), (13, 14), (14, 15), (15, 16), # Anular
        (13, 17), (17, 18), (18, 19), (19, 20),# Meñique
        (0, 17)                                # Base de la palma
    ]

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.last_detected_hands_count = 0

        # Detectar si estamos en la API clásica (mp.solutions) o en MediaPipe Tasks API
        self.is_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self._hands = None

        if self.is_legacy_solutions:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles
        else:
            # MediaPipe Tasks API moderno (0.10.x / 1.0+ / Python 3.13+)
            target_model_path = ensure_model_file(model_path or DEFAULT_MODEL_PATH)
            base_options = mp.tasks.BaseOptions(model_asset_path=target_model_path)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    @property
    def hands(self):
        """Retorna la instancia activa del estimador (Solutions Hands o Tasks Landmarker)."""
        return self._hands if self.is_legacy_solutions else self.landmarker

    @hands.setter
    def hands(self, value):
        self._hands = value

    def get_fingertip_points(
        self, frame: np.ndarray, draw_debug: bool = False
    ) -> np.ndarray:
        """
        Extrae las coordenadas en píxeles (X, Y) de las puntas de los dedos
        ([4, 8, 12, 16, 20]) de cada mano detectada.

        Args:
            frame: Imagen BGR donde proyectar las coordenadas.
            draw_debug: Si es True, dibuja el esqueleto y círculos sobre cada punta.

        Returns:
            np.ndarray con forma (N, 2) conteniendo las coordenadas [[x, y], ...].
        """
        height, width = frame.shape[:2]
        points: List[Tuple[int, int]] = []

        self.has_closed_hand = False
        self.closed_hands_count = 0

        if self.is_legacy_solutions:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            self.last_detected_hands_count = (
                len(results.multi_hand_landmarks)
                if results and results.multi_hand_landmarks
                else 0
            )

            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self._check_if_hand_closed(hand_landmarks):
                        self.has_closed_hand = True
                        self.closed_hands_count += 1

                    if draw_debug:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw_styles.get_default_hand_landmarks_style(),
                            self.mp_draw_styles.get_default_hand_connections_style(),
                        )

                    for tip_idx in self.FINGERTIP_INDICES:
                        lm = hand_landmarks.landmark[tip_idx]
                        cx = int(np.clip(lm.x * width, 0, width - 1))
                        cy = int(np.clip(lm.y * height, 0, height - 1))
                        points.append((cx, cy))
        else:
            # MediaPipe Tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.landmarker.detect(mp_image)

            self.last_detected_hands_count = (
                len(detection_result.hand_landmarks)
                if detection_result and detection_result.hand_landmarks
                else 0
            )

            if detection_result and detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    is_closed = self._check_if_hand_closed(hand_landmarks)
                    if is_closed:
                        self.has_closed_hand = True
                        self.closed_hands_count += 1

                    # Dibujar conexiones si se solicita vista de depuración
                    if draw_debug:
                        lm_pixel_map = {}
                        for i, lm in enumerate(hand_landmarks):
                            px = int(np.clip(lm.x * width, 0, width - 1))
                            py = int(np.clip(lm.y * height, 0, height - 1))
                            lm_pixel_map[i] = (px, py)

                        for start_idx, end_idx in self.HAND_CONNECTIONS:
                            if start_idx in lm_pixel_map and end_idx in lm_pixel_map:
                                cv2.line(
                                    frame,
                                    lm_pixel_map[start_idx],
                                    lm_pixel_map[end_idx],
                                    (0, 220, 255),
                                    2,
                                    cv2.LINE_AA,
                                )

                        if is_closed and len(hand_landmarks) > 0:
                            wx = int(np.clip(hand_landmarks[0].x * width, 0, width - 1))
                            wy = int(np.clip(hand_landmarks[0].y * height, 0, height - 1))
                            cv2.putText(
                                frame,
                                "PUNO CERRADO",
                                (wx - 40, max(20, wy - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )

                    # Extraer puntas de los dedos
                    for tip_idx in self.FINGERTIP_INDICES:
                        lm = hand_landmarks[tip_idx]
                        cx = int(np.clip(lm.x * width, 0, width - 1))
                        cy = int(np.clip(lm.y * height, 0, height - 1))
                        points.append((cx, cy))

                        if draw_debug:
                            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), cv2.FILLED)
                            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)

        return np.array(points, dtype=np.int32)

    def _check_if_hand_closed(self, hand_landmarks) -> bool:
        """
        Determina si una mano está cerrada (puño) verificando si al menos
        3 dedos principales tienen sus puntas dobladas hacia la muñeca.
        """
        is_list = isinstance(hand_landmarks, list)
        wrist = hand_landmarks[0] if is_list else hand_landmarks.landmark[0]
        wx, wy = wrist.x, wrist.y

        folded_count = 0
        # Pares (punta, articulación PIP): Índice (8,6), Medio (12,10), Anular (16,14), Meñique (20,18)
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_idx, pip_idx in finger_pairs:
            tip = hand_landmarks[tip_idx] if is_list else hand_landmarks.landmark[tip_idx]
            pip = hand_landmarks[pip_idx] if is_list else hand_landmarks.landmark[pip_idx]

            d_tip_sq = (tip.x - wx) ** 2 + (tip.y - wy) ** 2
            d_pip_sq = (pip.x - wx) ** 2 + (pip.y - wy) ** 2

            if d_tip_sq < d_pip_sq:
                folded_count += 1

        return folded_count >= 3

    def close(self) -> None:
        """Libera los recursos del modelo de MediaPipe."""
        if self.is_legacy_solutions and hasattr(self, "hands"):
            self.hands.close()
        elif hasattr(self, "landmarker"):
            self.landmarker.close()
