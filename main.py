"""
HandPortal - Aplicación Principal.
Orquesta la captura de video con selector ultrasencillo de cámaras (clic en pantalla,
barra espaciadora, tecla C o teclas numéricas 0/1), seguimiento de manos con MediaPipe,
geometría convexa, suavizado anti-flicker y renderizado dimensional.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.hand_tracker import HandTracker
from src.geometry import (
    compute_convex_hull,
    is_portal_activated,
    TemporalSmoother,
    calculate_polygon_area,
    FistGestureDetector,
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


def get_camera_backend() -> int:
    """Retorna el backend de captura óptimo para la plataforma."""
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def find_available_cameras(max_tested: int = 4) -> List[int]:
    """Escanea los índices de video disponibles en el sistema."""
    available: List[int] = []
    backend = get_camera_backend()
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(idx)
            cap.release()
    return available if available else [0]


def get_camera_label(camera_idx: int) -> str:
    """Devuelve una etiqueta descriptiva según el índice de la cámara."""
    if camera_idx == 1:
        return "Cámara USB Externa"
    elif camera_idx == 0:
        return "Cámara Integrada (Laptop)"
    else:
        return f"Cámara Externa/Aux ({camera_idx})"


def open_camera(camera_idx: int, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Abre y configura una cámara web usando DirectShow con bajo retardo."""
    backend = get_camera_backend()
    cap = cv2.VideoCapture(camera_idx, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


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
    camera_idx: int,
    total_cameras: int,
    has_closed_hand: bool = False,
) -> None:
    """Dibuja un HUD translúcido en pantalla con información de estado."""
    hud_h = 186
    hud_w = 385
    overlay = frame.copy()

    # Panel translúcido superior izquierdo
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + hud_w, 12 + hud_h), (80, 85, 105), 1)

    # Título
    cv2.putText(
        frame,
        "HAND PORTAL v1.3",
        (24, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    # Estado del portal
    status_text = "PORTAL: ABIERTO" if portal_open else "PORTAL: EN ESPERA"
    status_color = (0, 255, 120) if portal_open else (120, 130, 255)
    cv2.putText(
        frame,
        status_text,
        (225, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.42,
        status_color,
        1,
        cv2.LINE_AA,
    )

    cv2.line(frame, (24, 42), (12 + hud_w - 12, 42), (60, 65, 85), 1)

    line1 = f"FPS: {fps:.1f}  |  Manos: {hands_count}  |  Puntas: {points_count}"
    cv2.putText(
        frame, line1, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 225, 225), 1
    )

    cam_label = get_camera_label(camera_idx)
    line2 = f"Camara [{camera_idx}]: {cam_label}"
    cv2.putText(
        frame, line2, (24, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1
    )

    line3 = f"Efecto [M]: {mode_name.upper()}  |  Borde [B]: {color_name}"
    cv2.putText(
        frame, line3, (24, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1
    )

    # Estado del gesto de puño cerrado
    fist_text = "PUNO CERRADO (CAMBIA FILTRO)" if has_closed_hand else "MANO ABIERTA"
    fist_color = (0, 165, 255) if has_closed_hand else (120, 255, 160)
    line4 = f"Gesto: {fist_text}"
    cv2.putText(
        frame, line4, (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fist_color, 1
    )

    smooth_status = "ON" if smoothing_active else "OFF"
    debug_status = "ON" if debug_mode else "OFF"
    line5 = f"Suavizado [S]: {smooth_status}  |  Debug [D]: {debug_status}"
    cv2.putText(
        frame, line5, (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1
    )

    line6 = "Cierra la mano: Cambia Filtro | Espacio: Cam"
    cv2.putText(
        frame, line6, (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 220, 160), 1
    )


def draw_camera_button(frame: np.ndarray, current_cam_idx: int) -> Tuple[int, int, int, int]:
    """
    Dibuja un botón grande e interactivo en la esquina superior derecha
    para cambiar de cámara con un simple clic del ratón.
    """
    btn_w = 280
    btn_h = 50
    btn_x1 = frame.shape[1] - btn_w - 18
    btn_y1 = 14
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (btn_x1, btn_y1), (btn_x2, btn_y2), (25, 35, 48), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Borde brillante cyan/azul
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 215, 255), 2)

    # Icono y texto de acción
    cv2.putText(
        frame,
        "CAMBIAR A LA OTRA CAMARA",
        (btn_x1 + 22, btn_y1 + 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[ Clic aqui  |  ESPACIO  |  Tecla C ]",
        (btn_x1 + 20, btn_y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    return (btn_x1, btn_y1, btn_x2, btn_y2)


def draw_toast_notification(
    frame: np.ndarray, message: str, color: Tuple[int, int, int] = (0, 255, 150)
) -> None:
    """Dibuja un mensaje flotante central temporal cuando se cambia de cámara o de filtro."""
    h, w = frame.shape[:2]
    toast_w = max(460, len(message) * 11 + 50)
    toast_h = 60
    tx1 = (w - toast_w) // 2
    ty1 = h - toast_h - 40
    tx2 = tx1 + toast_w
    ty2 = ty1 + toast_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (15, 25, 35), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 2)

    cv2.putText(
        frame,
        message,
        (tx1 + 24, ty1 + 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HandPortal: Portal dimensional interactivo con OpenCV y MediaPipe."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Índice de cámara (por defecto selecciona automáticamente la cámara USB)",
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

    # Detección automática de cámaras disponibles
    print("[INFO] Escaneando dispositivos de cámara disponibles...")
    available_cameras = find_available_cameras(max_tested=4)
    print(f"[INFO] Cámaras detectadas: {available_cameras}")

    # Priorizar la cámara USB (índice 1) si existe
    if args.camera is not None:
        active_camera_idx = args.camera
    else:
        active_camera_idx = 1 if 1 in available_cameras else available_cameras[0]

    print(f"[INFO] Iniciando con [{active_camera_idx}]: {get_camera_label(active_camera_idx)}")

    # Inicializar captura
    cap = open_camera(active_camera_idx, args.width, args.height)
    if not cap.isOpened():
        fallback = available_cameras[0] if available_cameras else 0
        print(f"[AVISO] Fallback a cámara {fallback}...")
        active_camera_idx = fallback
        cap = open_camera(active_camera_idx, args.width, args.height)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir ninguna cámara.")
            sys.exit(1)

    # Componentes principales
    tracker = HandTracker(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6
    )
    smoother = TemporalSmoother(num_samples=36, alpha=0.35)
    video_loader = VideoTextureLoader(args.video)
    gesture_detector = FistGestureDetector(cooldown=0.75)

    # Estado de la aplicación
    modes = PORTAL_MODES
    current_mode_idx = modes.index(args.mode) if args.mode in modes else 0
    color_keys = list(BORDER_COLORS.keys())
    current_color_idx = 0
    show_hud = True
    debug_view = False
    enable_smoothing = not args.no_smooth

    # Control de cambio de cámara interactivo (por clic de ratón)
    request_camera_switch = False
    request_target_camera: int | None = None
    btn_bounds = (0, 0, 0, 0)

    def on_mouse_click(event, x, y, flags, param):
        nonlocal request_camera_switch, btn_bounds
        if event == cv2.EVENT_LBUTTONDOWN:
            bx1, by1, bx2, by2 = btn_bounds
            # Clic dentro del botón superior derecho
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                request_camera_switch = True
            # Clic dentro del renglón de cámara en el HUD (esquina sup. izquierda)
            elif 12 <= x <= 390 and 70 <= y <= 95:
                request_camera_switch = True

    window_name = "Hand Portal - MediaPipe & OpenCV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click)

    # Feedback visual (Toast)
    toast_message = f"Cámara: {get_camera_label(active_camera_idx)}"
    toast_color = (0, 255, 150)
    toast_until_time = time.time() + 2.5

    # Métricas
    prev_time = time.time()
    fps = 0.0
    tick_count = 0

    print("==================================================")
    print("  HAND PORTAL INICIADO CORRECTAMENTE")
    print("==================================================")
    print("  [PUÑO CERRADO] - Cierra la mano para cambiar el filtro al instante")
    print("  [CLIC RATON]   - Haz clic en el botón superior derecho para cambiar cámara")
    print("  [ESPACIO] / C  - Cambiar a la otra cámara al instante")
    print("  [0] / [1]      - Seleccionar cámara 0 (Interna) o 1 (USB)")
    print("  [M]            - Cambiar modo de efecto dimensional")
    print("  [B]            - Cambiar color del borde")
    print("  [S]            - Alternar suavizado temporal anti-flicker")
    print("  [D]            - Alternar vista de esqueleto / debug")
    print("  [H]            - Mostrar / ocultar HUD")
    print("  [Q] o [ESC]    - Salir")
    print("==================================================")

    try:
        while True:
            # Procesar solicitud de cambio de cámara si se activó por clic o teclado
            if request_camera_switch or request_target_camera is not None:
                if len(available_cameras) > 1 or request_target_camera is not None:
                    if request_target_camera is not None:
                        new_cam_idx = request_target_camera
                    else:
                        curr_pos = (
                            available_cameras.index(active_camera_idx)
                            if active_camera_idx in available_cameras
                            else 0
                        )
                        new_cam_idx = available_cameras[
                            (curr_pos + 1) % len(available_cameras)
                        ]

                    if new_cam_idx != active_camera_idx or not cap.isOpened():
                        print(f"\n[CAMBIANDO CÁMARA] -> [{new_cam_idx}]: {get_camera_label(new_cam_idx)}")
                        cap.release()
                        active_camera_idx = new_cam_idx
                        cap = open_camera(active_camera_idx, args.width, args.height)
                        smoother.reset()
                        toast_message = f"CAMBIADA A: [{active_camera_idx}] {get_camera_label(active_camera_idx)}"
                        toast_color = (0, 255, 150)
                        toast_until_time = time.time() + 2.5

                request_camera_switch = False
                request_target_camera = None

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 1. Efecto espejo horizontal
            frame = cv2.flip(frame, 1)
            tick_count += 1

            # 2. Extracción de coordenadas de las puntas de los dedos
            points = tracker.get_fingertip_points(frame, draw_debug=debug_view)
            hands_count = tracker.last_detected_hands_count
            points_count = len(points)

            # 3. Gesto de puño cerrado: cambiar el filtro dimensional automáticamente
            if gesture_detector.update(tracker.has_closed_hand):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                current_mode = modes[current_mode_idx]
                toast_message = f"PUNO CERRADO -> FILTRO: {current_mode.upper()}"
                toast_color = (0, 215, 255)
                toast_until_time = time.time() + 2.0
                print(f"\n[GESTO DE PUÑO]: Mano cerrada -> Filtro cambiado a: {current_mode.upper()}")

            # 4. Comprobar umbral de activación
            portal_open = is_portal_activated(points, min_points=3, min_area=3000.0)

            current_mode = modes[current_mode_idx]
            current_color_name = color_keys[current_color_idx]
            current_border_color = BORDER_COLORS[current_color_name]

            # 5. Renderizado del portal
            if portal_open:
                hull = compute_convex_hull(points)
                if len(hull) >= 3:
                    if enable_smoothing:
                        render_polygon = smoother.smooth(hull)
                    else:
                        render_polygon = hull

                    external_frame = (
                        video_loader.get_frame() if current_mode == "video" else None
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
                smoother.reset()

            # 6. FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 7. Dibujar botón interactivo superior derecho para cambiar cámara
            btn_bounds = draw_camera_button(frame, active_camera_idx)

            # 8. Dibujar HUD
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
                    camera_idx=active_camera_idx,
                    total_cameras=len(available_cameras),
                    has_closed_hand=tracker.has_closed_hand,
                )

            # 9. Notificación flotante de confirmación de cambio de cámara o filtro
            if time.time() < toast_until_time:
                draw_toast_notification(frame, toast_message, color=toast_color)

            cv2.imshow(window_name, frame)

            # 9. Gestión de eventos de teclado
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' o ESC
                break
            # BARRA ESPACIADORA (32), Tecla 'C', 'N' o TAB (9) -> Cambiar a la otra cámara
            elif key in (32, ord("c"), ord("C"), ord("n"), ord("N"), 9):
                request_camera_switch = True
            # Teclas numéricas '0', '1', '2' -> Seleccionar cámara directamente
            elif key in (ord("0"), ord("1"), ord("2")):
                target = int(chr(key))
                if target in available_cameras:
                    request_target_camera = target
            elif key in (ord("m"), ord("M")):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                print(f"[MODO CAMBIADO]: {modes[current_mode_idx]}")
            elif key in (ord("b"), ord("B")):  # B de Borde
                current_color_idx = (current_color_idx + 1) % len(color_keys)
                print(f"[COLOR DE BORDE]: {color_keys[current_color_idx]}")
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
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados y aplicación finalizada.")


if __name__ == "__main__":
    main()
