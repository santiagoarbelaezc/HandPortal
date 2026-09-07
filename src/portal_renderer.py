"""
Módulo de renderizado y fusión de capas para el efecto de portal dimensional.
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np

PORTAL_MODES = ["magma", "inferno", "invert", "cosmic", "video"]


def generate_portal_content(
    frame: np.ndarray,
    mode: str = "magma",
    external_frame: Optional[np.ndarray] = None,
    tick: int = 0,
) -> np.ndarray:
    """
    Genera el contenido visual alternativo proyectado en el interior del portal.

    Args:
        frame: Cuadro de video actual capturado por la cámara.
        mode: Modo visual ('magma', 'inferno', 'invert', 'cosmic', 'video').
        external_frame: Cuadro de video o textura alternativa (opcional).
        tick: Contador de frames para animaciones procedurales dinámicas.

    Returns:
        Frame transformado del mismo tamaño que el original.
    """
    h, w = frame.shape[:2]

    if mode == "video" and external_frame is not None:
        if external_frame.shape[:2] != (h, w):
            return cv2.resize(external_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return external_frame

    if mode == "magma":
        return cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

    if mode == "inferno":
        return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)

    if mode == "invert":
        return cv2.bitwise_not(frame)

    if mode == "cosmic":
        # Transformación espacial en HSV con rotación de tono y detección de bordes Neón
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Rotar tono cíclicamente con el tiempo
        hsv[..., 0] = (hsv[..., 0] + (tick * 2) % 180) % 180
        # Saturar al máximo
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
        cosmic_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Detectar bordes de Laplace y sumarlos para un look hiperespacial
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)
        return cv2.addWeighted(cosmic_bgr, 0.75, edges_colored, 0.45, 0)

    # Por defecto 'magma'
    return cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)


def apply_portal_effect(
    frame: np.ndarray,
    points: np.ndarray,
    mode: str = "magma",
    external_frame: Optional[np.ndarray] = None,
    border_color: Tuple[int, int, int] = (0, 0, 255),
    border_thickness: int = 2,
    feather_edges: bool = True,
    tick: int = 0,
) -> np.ndarray:
    """
    Crea la máscara binaria, genera la capa del portal, fusiona ambas capas
    y dibuja los bordes luminosos sobre el frame.

    Args:
        frame: Imagen BGR de la cámara.
        points: Coordenadas de los vértices del portal (N, 2).
        mode: Modo visual del interior ('magma', 'inferno', 'invert', 'cosmic', 'video').
        external_frame: Cuadro de video o textura opcional.
        border_color: Color BGR para el borde del polígono (por defecto Rojo: (0, 0, 255)).
        border_thickness: Grosor de línea del borde del portal.
        feather_edges: Si es True, suaviza suavemente el contorno de fusión con desenfoque.
        tick: Contador de frames para efectos animados.

    Returns:
        Frame procesado con el portal renderizado.
    """
    if points is None or len(points) < 3:
        return frame

    h, w = frame.shape[:2]
    polygon = points.astype(np.int32)

    # 1. Generar máscara binaria negra del tamaño del frame
    mask = np.zeros((h, w), dtype=np.uint8)

    # 2. Dibujar el polígono en blanco
    cv2.fillPoly(mask, [polygon], 255)

    # 3. Generar la capa de contenido del portal
    effect_frame = generate_portal_content(
        frame=frame,
        mode=mode,
        external_frame=external_frame,
        tick=tick,
    )

    # 4. Fusión de capas (blending)
    if feather_edges:
        # Suavizado suave de bordes para una fusión fotográfica limpia
        blur_mask = cv2.GaussianBlur(mask, (9, 9), 0)
        norm_mask = (blur_mask.astype(np.float32) / 255.0)[..., np.newaxis]
        blended = (effect_frame.astype(np.float32) * norm_mask) + (
            frame.astype(np.float32) * (1.0 - norm_mask)
        )
        result = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # Fusión binaria directa con np.where
        mask_3ch = mask[..., np.newaxis] == 255
        result = np.where(mask_3ch, effect_frame, frame)

    # 5. Trazar borde exterior (brillo / halo) y contorno nítido
    # Halo difuminado exterior
    glow_color = tuple(min(255, int(c * 0.6)) for c in border_color)
    cv2.polylines(
        result,
        [polygon],
        isClosed=True,
        color=glow_color,
        thickness=border_thickness + 3,
        lineType=cv2.LINE_AA,
    )

    # Borde principal nítido
    cv2.polylines(
        result,
        [polygon],
        isClosed=True,
        color=border_color,
        thickness=border_thickness,
        lineType=cv2.LINE_AA,
    )

    return result
