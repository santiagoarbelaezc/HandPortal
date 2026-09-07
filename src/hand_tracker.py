"""
Módulo de detección y extracción de landmarks de manos utilizando MediaPipe.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp


class HandTracker:
    """
    Encapsula el modelo MediaPipe Hands para detectar manos en tiempo real
    y extraer las coordenadas de las puntas de los dedos.
    """

    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Pulgar, Índice, Medio, Anular, Meñique

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Inicializa la instancia de MediaPipe Hands.

        Args:
            max_num_hands: Número máximo de manos a detectar simultáneamente.
            min_detection_confidence: Umbral de confianza mínimo para detección inicial.
            min_tracking_confidence: Umbral de confianza mínimo para seguimiento de landmarks.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        self.results = None
        self.last_detected_hands_count = 0

    def find_hands(self, frame: np.ndarray, draw: bool = False) -> np.ndarray:
        """
        Procesa el frame BGR, ejecuta la inferencia en RGB y opcionalmente dibuja los landmarks.

        Args:
            frame: Imagen BGR capturada desde la cámara.
            draw: Booleano para dibujar los esqueletos de las manos detectadas.

        Returns:
            Frame con o sin dibujos de landmarks.
        """
        # MediaPipe requiere formato de color RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        self.last_detected_hands_count = (
            len(self.results.multi_hand_landmarks)
            if self.results and self.results.multi_hand_landmarks
            else 0
        )

        if draw and self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style(),
                )

        return frame

    def get_fingertip_points(
        self, frame: np.ndarray, draw_debug: bool = False
    ) -> np.ndarray:
        """
        Extrae las coordenadas en píxeles (X, Y) de las puntas de los dedos
        ([4, 8, 12, 16, 20]) de cada mano detectada.

        Args:
            frame: Imagen BGR donde proyectar las coordenadas.
            draw_debug: Si es True, dibuja círculos sobre cada punta detectada.

        Returns:
            np.ndarray de forma (N, 2) con formato [[x, y], ...] en coordenadas de píxeles.
        """
        self.find_hands(frame, draw=draw_debug)

        points: List[Tuple[int, int]] = []
        if not self.results or not self.results.multi_hand_landmarks:
            return np.empty((0, 2), dtype=np.int32)

        height, width, _ = frame.shape

        for hand_landmarks in self.results.multi_hand_landmarks:
            for tip_idx in self.FINGERTIP_INDICES:
                lm = hand_landmarks.landmark[tip_idx]
                cx, cy = int(lm.x * width), int(lm.y * height)
                # Restringir las coordenadas al rango válido de la imagen
                cx = max(0, min(width - 1, cx))
                cy = max(0, min(height - 1, cy))
                points.append((cx, cy))

                if draw_debug:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 8, (0, 165, 255), 1)

        return np.array(points, dtype=np.int32)

    def close(self) -> None:
        """Libera los recursos del grafo de MediaPipe."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
