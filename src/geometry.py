"""
Módulo de geometría: cálculo de polígonos, envolvente convexa, orden polar,
distancias y suavizado temporal contra parpadeos.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def compute_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Calcula la envolvente convexa (Convex Hull) de un conjunto de puntos 2D.

    Args:
        points: Array NumPy con forma (N, 2) de coordenadas en píxeles.

    Returns:
        Array NumPy con forma (M, 2) conteniendo los vértices de la envolvente convexa.
    """
    if points is None or len(points) < 3:
        return np.empty((0, 2), dtype=np.int32)

    pts = points.astype(np.int32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)


def order_points_polar(points: np.ndarray) -> np.ndarray:
    """
    Ordena un conjunto de puntos según su ángulo polar respecto al centroide.
    Evita que las líneas del polígono se crucen o auto-intersequen.

    Args:
        points: Array NumPy con forma (N, 2).

    Returns:
        Array NumPy con forma (N, 2) ordenado angularmente en sentido horario.
    """
    if points is None or len(points) < 3:
        return points if points is not None else np.empty((0, 2), dtype=np.int32)

    cx = float(np.mean(points[:, 0]))
    cy = float(np.mean(points[:, 1]))

    # Ángulo en radianes respecto al centroide [-pi, pi]
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    sorted_indices = np.argsort(angles)
    return points[sorted_indices].astype(np.int32)


def calculate_polygon_area(polygon: np.ndarray) -> float:
    """
    Calcula el área del polígono en píxeles cuadrados.

    Args:
        polygon: Array con forma (N, 2) o (N, 1, 2).

    Returns:
        Área como valor flotante.
    """
    if polygon is None or len(polygon) < 3:
        return 0.0
    return float(cv2.contourArea(polygon.astype(np.int32)))


def calculate_centroid(points: np.ndarray) -> Tuple[int, int]:
    """
    Calcula el centroide (cx, cy) de un conjunto de puntos.
    """
    if points is None or len(points) == 0:
        return (0, 0)
    cx = int(np.mean(points[:, 0]))
    cy = int(np.mean(points[:, 1]))
    return (cx, cy)


def is_portal_activated(
    points: np.ndarray,
    min_points: int = 4,
    min_area: float = 3500.0,
) -> bool:
    """
    Verifica los umbrales de activación para abrir el portal:
    número mínimo de puntos detectados y área mínima del contorno.
    """
    if points is None or len(points) < min_points:
        return False

    hull = compute_convex_hull(points)
    if len(hull) < 3:
        return False

    area = calculate_polygon_area(hull)
    return area >= min_area


class TemporalSmoother:
    """
    Suavizador temporal para contornos de polígonos utilizando
    interpolación lineal ponderada (LERP / Filtro de Media Móvil Exponencial).
    Normaliza el polígono a un número fijo de rayos radiales desde el centroide
    para evitar saltos dimensionales cuando varía la cantidad de vértices detectados.
    """

    def __init__(self, num_samples: int = 36, alpha: float = 0.35) -> None:
        """
        Args:
            num_samples: Cantidad de puntos angulares muestreados (p. ej. 36 = cada 10°).
            alpha: Factor de suavizado LERP (0.0 = congelado, 1.0 = sin suavizado / instantáneo).
        """
        self.num_samples = num_samples
        self.alpha = np.clip(alpha, 0.05, 1.0)
        self.angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        self.cached_radii: Optional[np.ndarray] = None
        self.cached_center: Optional[np.ndarray] = None
        self.active = False

    def reset(self) -> None:
        """Reinicia el historial de suavizado."""
        self.cached_radii = None
        self.cached_center = None
        self.active = False

    def smooth(self, hull_points: np.ndarray) -> np.ndarray:
        """
        Aplica suavizado temporal sobre el contorno dado.

        Args:
            hull_points: Vértices de la envolvente convexa actual (M, 2).

        Returns:
            Polígono suavizado con forma (num_samples, 2).
        """
        if hull_points is None or len(hull_points) < 3:
            self.reset()
            return np.empty((0, 2), dtype=np.int32)

        # Centroide actual
        current_cx = float(np.mean(hull_points[:, 0]))
        current_cy = float(np.mean(hull_points[:, 1]))
        current_center = np.array([current_cx, current_cy], dtype=np.float32)

        # Calcular distancias radiales del contorno hacia los rayos angulares
        current_radii = self._sample_radii(hull_points, current_center)

        if not self.active or self.cached_radii is None or self.cached_center is None:
            # Primer frame válido
            self.cached_radii = current_radii
            self.cached_center = current_center
            self.active = True
        else:
            # LERP / Exponential Moving Average
            self.cached_radii = (
                self.alpha * current_radii + (1.0 - self.alpha) * self.cached_radii
            )
            self.cached_center = (
                self.alpha * current_center + (1.0 - self.alpha) * self.cached_center
            )

        # Reconstruir coordenadas cartesianas a partir de radios y centro suavizado
        xs = self.cached_center[0] + self.cached_radii * np.cos(self.angles)
        ys = self.cached_center[1] + self.cached_radii * np.sin(self.angles)

        smooth_polygon = np.column_stack((xs, ys)).astype(np.int32)
        return smooth_polygon

    def _sample_radii(self, polygon: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Calcula el radio proyectado del polígono para cada rayo angular.
        """
        # Calcular vectores y ángulos de todos los vértices del polígono respecto al centro
        diffs = polygon.astype(np.float32) - center
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        angles = np.arctan2(diffs[:, 1], diffs[:, 0]) % (2 * np.pi)

        # Ordenar por ángulo
        order = np.argsort(angles)
        angles_sorted = angles[order]
        dists_sorted = dists[order]

        # Interpolar distancias en los ángulos uniformes precalculados
        # Duplicar ciclo para manejar periodicidad circular continua [0, 2*pi]
        angles_extended = np.concatenate(
            [angles_sorted - 2 * np.pi, angles_sorted, angles_sorted + 2 * np.pi]
        )
        dists_extended = np.concatenate(
            [dists_sorted, dists_sorted, dists_sorted]
        )

        radii = np.interp(self.angles, angles_extended, dists_extended)
        return radii
