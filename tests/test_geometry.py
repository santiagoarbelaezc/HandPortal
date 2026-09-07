"""
Pruebas de verificación automatizadas para módulos de HandPortal:
Geometría, Envolvente Convexa, Suavizado Temporal y Renderizado de Portal.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry import (
    compute_convex_hull,
    order_points_polar,
    calculate_polygon_area,
    calculate_centroid,
    is_portal_activated,
    TemporalSmoother,
)
from src.portal_renderer import (
    apply_portal_effect,
    generate_portal_content,
    PORTAL_MODES,
)
from src.hand_tracker import HandTracker


class TestHandPortalModules(unittest.TestCase):

    def setUp(self):
        # Crear un frame sintético de prueba (720p, 3 canales)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(self.frame, (100, 100), (400, 400), (120, 180, 240), -1)

        # 8 puntos simulando las puntas de los dedos de 2 manos formando un óvalo/portal
        self.sample_points = np.array(
            [
                [400, 200],
                [600, 150],
                [800, 220],
                [900, 350],
                [850, 500],
                [650, 550],
                [450, 480],
                [350, 340],
            ],
            dtype=np.int32,
        )

    def test_convex_hull(self):
        hull = compute_convex_hull(self.sample_points)
        self.assertGreaterEqual(len(hull), 3)
        self.assertEqual(hull.ndim, 2)
        self.assertEqual(hull.shape[1], 2)

    def test_order_points_polar(self):
        ordered = order_points_polar(self.sample_points)
        self.assertEqual(len(ordered), len(self.sample_points))
        centroid = calculate_centroid(ordered)
        self.assertGreater(centroid[0], 0)
        self.assertGreater(centroid[1], 0)

    def test_polygon_area_and_activation(self):
        hull = compute_convex_hull(self.sample_points)
        area = calculate_polygon_area(hull)
        self.assertGreater(area, 50000.0)
        self.assertTrue(is_portal_activated(self.sample_points, min_points=4, min_area=3000.0))

    def test_temporal_smoother(self):
        smoother = TemporalSmoother(num_samples=36, alpha=0.4)
        hull = compute_convex_hull(self.sample_points)

        # Frame 1
        smoothed_1 = smoother.smooth(hull)
        self.assertEqual(smoothed_1.shape, (36, 2))

        # Frame 2 con ligera variación sintética
        perturbed_points = self.sample_points + np.random.randint(-5, 5, size=self.sample_points.shape)
        hull_2 = compute_convex_hull(perturbed_points)
        smoothed_2 = smoother.smooth(hull_2)
        self.assertEqual(smoothed_2.shape, (36, 2))

    def test_portal_renderer_all_modes(self):
        hull = compute_convex_hull(self.sample_points)
        ext_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 150

        for mode in PORTAL_MODES:
            rendered = apply_portal_effect(
                frame=self.frame.copy(),
                points=hull,
                mode=mode,
                external_frame=ext_frame,
                border_color=(0, 0, 255),
                border_thickness=2,
            )
            self.assertEqual(rendered.shape, self.frame.shape)
            self.assertEqual(rendered.dtype, np.uint8)

    def test_hand_tracker_init(self):
        tracker = HandTracker(max_num_hands=2)
        self.assertIsNotNone(tracker.hands)
        tracker.close()


if __name__ == "__main__":
    unittest.main()
