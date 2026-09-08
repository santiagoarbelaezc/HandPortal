"""
Módulo de renderizado y fusión de capas para el efecto de portal dimensional.
Incluye una amplia colección de filtros dimensionales visuales (Magma, Inferno, Plasma,
Matrix, Cyberpunk, Glitch Multiverso, Ocean, Twilight, Rainbow, Invert y Video).
"""

from typing import Optional, Tuple, Dict
import cv2
import numpy as np

PORTAL_MODES = [
    "magma",
    "inferno",
    "plasma",
    "matrix",
    "cyberpunk",
    "glitch",
    "ocean",
    "twilight",
    "rainbow",
    "invert",
    "video",
]


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
        mode: Modo visual activo.
        external_frame: Cuadro de video o textura alternativa (opcional).
        tick: Contador de frames para animaciones procedurales dinámicas.

    Returns:
        Frame transformado del mismo tamaño que el original.
    """
    h, w = frame.shape[:2]

    # 1. Modo Video externo
    if mode == "video" and external_frame is not None:
        if external_frame.shape[:2] != (h, w):
            return cv2.resize(external_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return external_frame

    # 2. Modos térmicos y energéticos
    if mode == "magma":
        return cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

    if mode == "inferno":
        return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)

    if mode == "plasma":
        return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)

    # 3. Modo Terminal Matrix (Fósforo verde de alto contraste con scanlines)
    if mode == "matrix":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        matrix_bgr = np.zeros_like(frame)
        matrix_bgr[..., 1] = gray_eq
        matrix_bgr[..., 0] = (gray_eq * 0.15).astype(np.uint8)
        # Líneas de barrido CRT
        matrix_bgr[::3, :] = (matrix_bgr[::3, :] * 0.65).astype(np.uint8)
        return matrix_bgr

    # 4. Modo Cyberpunk (Duotono Neón Cian / Magenta de alto impacto)
    if mode == "cyberpunk":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        cyber = np.zeros_like(frame)
        cyber[..., 0] = np.clip((1.0 - gray) * 240.0 + gray * 40.0, 0, 255).astype(np.uint8)  # Azul/Cian
        cyber[..., 1] = np.clip(gray * 25.0, 0, 255).astype(np.uint8)                         # Verde
        cyber[..., 2] = np.clip(gray * 255.0, 0, 255).astype(np.uint8)                        # Rojo/Magenta
        return cyber

    # 5. Modo Glitch Multiverso (Aberración cromática horizontal + interferencia VHS)
    if mode == "glitch":
        shift = int(12 + 6 * np.sin(tick * 0.25))
        b, g, r = cv2.split(frame)
        b_shifted = np.roll(b, shift, axis=1)
        r_shifted = np.roll(r, -shift, axis=1)
        glitched = cv2.merge([b_shifted, g, r_shifted])
        # Scanlines horizontales
        glitched[::4, :] = (glitched[::4, :] * 0.70).astype(np.uint8)
        return glitched

    # 6. Modo Océano Abisal (Bioluminiscencia marina)
    if mode == "ocean":
        return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)

    # 7. Modo Twilight (Crepúsculo estelar)
    if mode == "twilight":
        return cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT_SHIFTED)

    # 8. Modo Rainbow / Espectral (Prisma dimensional continuo)
    if mode == "rainbow":
        return cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)

    # 9. Modo Invert (Negativo cuántico espectral)
    if mode == "invert":
        return cv2.bitwise_not(frame)

    # Fallback por defecto a magma
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
    Crea la máscara binaria, genera la capa del portal con el filtro seleccionado,
    fusiona ambas capas y dibuja los bordes luminosos sobre el frame.

    Args:
        frame: Imagen BGR de la cámara.
        points: Coordenadas de los vértices del portal (N, 2).
        mode: Modo visual del interior.
        external_frame: Cuadro de video o textura opcional.
        border_color: Color BGR para el borde del polígono.
        border_thickness: Grosor de línea del borde del portal.
        feather_edges: Si es True, suaviza el contorno de fusión con desenfoque Gaussiano.
        tick: Contador de frames para efectos dinámicos.

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
        blur_mask = cv2.GaussianBlur(mask, (9, 9), 0)
        norm_mask = (blur_mask.astype(np.float32) / 255.0)[..., np.newaxis]
        blended = (effect_frame.astype(np.float32) * norm_mask) + (
            frame.astype(np.float32) * (1.0 - norm_mask)
        )
        result = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        mask_3ch = mask[..., np.newaxis] == 255
        result = np.where(mask_3ch, effect_frame, frame)

    # 5. Trazar halo difuminado exterior y contorno nítido
    glow_color = tuple(min(255, int(c * 0.6)) for c in border_color)
    cv2.polylines(
        result,
        [polygon],
        isClosed=True,
        color=glow_color,
        thickness=border_thickness + 3,
        lineType=cv2.LINE_AA,
    )

    cv2.polylines(
        result,
        [polygon],
        isClosed=True,
        color=border_color,
        thickness=border_thickness,
        lineType=cv2.LINE_AA,
    )

    return result
