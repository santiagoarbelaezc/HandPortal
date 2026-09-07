"""
HandPortal - Aplicación Principal.
Orquesta la captura de video, el seguimiento de manos con MediaPipe,
la geometría convexa, el suavizado temporal y el renderizado dimensional del portal.
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from src.hand_tracker import HandTracker
from src.geometry import (
    compute_convex_hull,
    is_portal_activated,
    TemporalSmoother,
    calculate_polygon_area,
)
from src.portal_renderer import (
    apply_portal_effect,
    generate_portal_content,
    PORTAL_MODES,
)

BORDER_COLORS = {
    "Rojo Portal": (0, 0, 255),
    "Cian Cuántico": (255, 230, 0),
    "Púrpura Cósmico": (255, 0, 180),
    "Verde Neón": (0, 255, 100),
    "Dorado Dimensional": (0, 215, 255),
}


class VideoTextureLoader:
    """Gestiona la lectura en bucle de un video externo para la textura del portal."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = None
        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)

    def get_frame(self) -> np.ndarray | None:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Reiniciar al inicio para reproducción en loop continuo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        return frame if ret else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def draw_hud(
    frame: np.ndarray,
    fps: float,
    hands_count: int,
    points_count: int,
    mode_name: str,
    color_name: str,
    smoothing_active: bool,
    debug_mode: bool,
    portal_open: bool,
) -> None:
    """Dibuja un HUD estilizado y translúcido en pantalla con información de estado."""
    hud_h = 135
    hud_w = 340
    overlay = frame.copy()

    # Panel translúcido superior izquierdo
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + hud_w, 12 + hud_h), (80, 80, 100), 1)

    # Título
    cv2.putText(
        frame,
        "HAND PORTAL v1.0",
        (24, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    # Estado del portal
    status_text = "PORTAL: ABIERTO" if portal_open else "PORTAL: EN ESPERA"
    status_color = (0, 255, 120) if portal_open else (100, 100, 255)
    cv2.putText(
        frame,
        status_text,
        (200, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.42,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Línea divisoria
    cv2.line(frame, (24, 42), (12 + hud_w - 12, 42), (60, 60, 80), 1)

    # Métricas
    line1 = f"FPS: {fps:.1f}  |  Manos: {hands_count}  |  Puntas: {points_count}"
    cv2.putText(
        frame, line1, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
    )

    line2 = f"Efecto [M]: {mode_name.upper()}  |  Color [C]: {color_name}"
    cv2.putText(
        frame, line2, (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
    )

    smooth_status = "ON" if smoothing_active else "OFF"
    debug_status = "ON" if debug_mode else "OFF"
    line3 = f"Suavizado [S]: {smooth_status}  |  Debug [D]: {debug_status}"
    cv2.putText(
        frame, line3, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1
    )

    line4 = "Presiona 'H': Ocultar HUD  |  'Q': Salir"
    cv2.putText(
        frame, line4, (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HandPortal: Portal dimensional interactivo con OpenCV y MediaPipe."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Índice de la cámara web (default: 0)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="magma",
        choices=PORTAL_MODES,
        help="Modo visual inicial del portal",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/portal_texture.mp4",
        help="Ruta a video de textura externa",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Ancho de captura deseado"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Alto de captura deseado"
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Desactiva el filtro de suavizado"
    )
    args = parser.parse_args()

    # Inicializar captura de video
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo acceder a la cámara con índice {args.camera}.")
        print("Intenta conectar una cámara o cambiar el índice con --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Componentes principales
    tracker = HandTracker(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6
    )
    smoother = TemporalSmoother(num_samples=36, alpha=0.35)
    video_loader = VideoTextureLoader(args.video)

    # Estado de la aplicación
    modes = PORTAL_MODES
    current_mode_idx = modes.index(args.mode) if args.mode in modes else 0
    color_keys = list(BORDER_COLORS.keys())
    current_color_idx = 0
    show_hud = True
    debug_view = False
    enable_smoothing = not args.no_smooth

    # Control de FPS
    prev_time = time.time()
    fps = 0.0
    tick_count = 0

    window_name = "Hand Portal - MediaPipe & OpenCV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("==================================================")
    print("  HAND PORTAL INICIADO CORRECTAMENTE")
    print("==================================================")
    print("  [Q]       - Salir")
    print("  [M]       - Alternar modo de efecto del portal")
    print("  [C]       - Alternar color del borde")
    print("  [S]       - Alternar suavizado temporal (anti-flicker)")
    print("  [D]       - Alternar vista de esqueleto / debug")
    print("  [H]       - Mostrar/ocultar panel HUD")
    print("==================================================")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] No se pudo leer cuadro de la cámara.")
                break

            # 1. Efecto espejo horizontal para interacción natural
            frame = cv2.flip(frame, 1)
            tick_count += 1

            # 2. Extracción de coordenadas de las puntas de los dedos
            points = tracker.get_fingertip_points(frame, draw_debug=debug_view)
            hands_count = tracker.last_detected_hands_count
            points_count = len(points)

            # 3. Comprobar umbral de activación del portal
            portal_open = is_portal_activated(
                points, min_points=3, min_area=3000.0
            )

            current_mode = modes[current_mode_idx]
            current_color_name = color_keys[current_color_idx]
            current_border_color = BORDER_COLORS[current_color_name]

            # 4. Renderizado del portal si se activa el umbral
            if portal_open:
                # Calcular la envolvente convexa
                hull = compute_convex_hull(points)

                if len(hull) >= 3:
                    if enable_smoothing:
                        render_polygon = smoother.smooth(hull)
                    else:
                        render_polygon = hull

                    external_frame = (
                        video_loader.get_frame()
                        if current_mode == "video"
                        else None
                    )

                    frame = apply_portal_effect(
                        frame=frame,
                        points=render_polygon,
                        mode=current_mode,
                        external_frame=external_frame,
                        border_color=current_border_color,
                        border_thickness=3,
                        feather_edges=True,
                        tick=tick_count,
                    )
            else:
                # Si se cierra el portal, reiniciar memoria del suavizador
                smoother.reset()

            # 5. Cálculo de FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 6. Dibujar HUD informativo
            if show_hud:
                draw_hud(
                    frame=frame,
                    fps=fps,
                    hands_count=hands_count,
                    points_count=points_count,
                    mode_name=current_mode,
                    color_name=current_color_name,
                    smoothing_active=enable_smoothing,
                    debug_mode=debug_view,
                    portal_open=portal_open,
                )

            cv2.imshow(window_name, frame)

            # 7. Gestión de eventos de teclado
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' o ESC
                break
            elif key in (ord("m"), ord("M")):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                print(f"[MODO CAMBIADO]: {modes[current_mode_idx]}")
            elif key in (ord("c"), ord("C")):
                current_color_idx = (current_color_idx + 1) % len(color_keys)
                print(f"[COLOR CAMBIADO]: {color_keys[current_color_idx]}")
            elif key in (ord("s"), ord("S")):
                enable_smoothing = not enable_smoothing
                if not enable_smoothing:
                    smoother.reset()
                print(f"[SUAVIZADO]: {'ACTIVADO' if enable_smoothing else 'DESACTIVADO'}")
            elif key in (ord("d"), ord("D")):
                debug_view = not debug_view
                print(f"[DEBUG]: {'ACTIVADO' if debug_view else 'DESACTIVADO'}")
            elif key in (ord("h"), ord("H")):
                show_hud = not show_hud

    finally:
        tracker.close()
        video_loader.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados y aplicación finalizada.")


if __name__ == "__main__":
    main()
