"""Exercise the optional notebook renderer, including its actual GPU buffers."""

import asyncio
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np
import quantities as q
from numpy.testing import assert_allclose

from coordsys import CoordinateSystem


@unittest.skipUnless(
    importlib.util.find_spec('fastplotlib'), 'optional notebook dependencies unavailable'
)
class GeometryViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Select the headless backend before fastplotlib imports rendercanvas.auto.
        with patch.dict(os.environ, RENDERCANVAS_BACKEND='offscreen'):
            from rendercanvas.offscreen import RenderCanvas

            from examples.geometry_viewer import GeometryViewer, NotebookCanvas

        cls.Viewer, cls.Canvas, cls.NotebookCanvas = GeometryViewer, RenderCanvas, NotebookCanvas

    def make_viewer(self):
        reference = CoordinateSystem()
        main = CoordinateSystem([100, 0, 0], unit=q.cm)
        main.rotate([0, 0, 1], 90 * q.deg)
        main.scale(2)
        child = CoordinateSystem([1000, 0, 0], parent=main, unit=q.mm)
        points = np.array([[1000, 0, 0], [0, 1000, 0]]) * q.mm
        child.add_points(points, 'cloud')
        viewer = self.Viewer(points.rescale(q.m).magnitude, canvas=self.Canvas(size=(640, 480)))
        self.addCleanup(viewer.close)
        viewer.update(reference, main, child)
        return viewer, reference, main, child

    @staticmethod
    def rendered_points(viewer):
        local = viewer.cloud.data[:]
        matrix = viewer.cloud.world_object.world.matrix
        return local @ matrix[:3, :3].T + matrix[:3, 3]

    def test_rendered_points_and_arrow_endpoints_with_mixed_units_and_scale(self):
        viewer, _, _, _ = self.make_viewer()
        assert_allclose(self.rendered_points(viewer), [[1, 4, 0], [-1, 2, 0]], atol=2e-6)
        mesh = viewer.arrows[2].world_object
        vertices = mesh.geometry.positions.data
        tail = [0, 0, vertices[:, 2].min(), 1]
        tip = [0, 0, vertices[:, 2].max(), 1]
        for i, direction in enumerate(([0, 2, 0], [-2, 0, 0], [0, 0, 2])):
            matrix = mesh.world.matrix @ mesh.get_matrix_at(i)
            assert_allclose((matrix @ tail)[:3], [1, 2, 0], atol=2e-6)
            assert_allclose((matrix @ tip)[:3], np.array([1, 2, 0]) + direction, atol=2e-6)
            # The arrow keeps a short head and a circular cross-section.
            length = np.linalg.norm(direction)
            relative = vertices @ matrix[:3, :3].T
            axial = relative @ (direction / length)
            radial = np.linalg.norm(relative - axial[:, None] * (direction / length), axis=1)
            assert_allclose(radial.max(), 0.045 * length, atol=2e-6)
            rim = radial > 0.99 * radial.max()
            assert_allclose(axial[rim], 0.82 * length, atol=2e-6)

    def test_updates_preserve_camera_and_geometry_with_inherit_false(self):
        viewer, reference, main, child = self.make_viewer()
        viewer.plot.camera.local.position = viewer.plot.camera.local.position + (1, 2, 3)
        viewer.plot.camera.zoom = 1.5
        camera = viewer.plot.camera.local.matrix.copy()
        projection = viewer.plot.camera.projection_matrix.copy()
        graphics = tuple(viewer.plot.graphics)
        points = self.rendered_points(viewer).copy()
        for _ in range(3):
            child.translate([100, 200, 300], inherit=False)
            viewer.update(reference, main, child)
            assert_allclose(self.rendered_points(viewer), points, atol=2e-6)
            assert_allclose(viewer.plot.camera.local.matrix, camera)
            assert_allclose(viewer.plot.camera.projection_matrix, projection)
            self.assertEqual(tuple(viewer.plot.graphics), graphics)
        viewer.reset_view()
        self.assertFalse(np.allclose(viewer.plot.camera.local.matrix, camera))
        self.assertEqual(viewer.plot.camera.zoom, 1)

    def test_offscreen_draw_has_white_background_and_screen_facing_labels(self):
        viewer, _, _, _ = self.make_viewer()
        for labels in viewer.labels:
            for label in labels:
                self.assertTrue(label.world_object.screen_space)
                self.assertFalse(label.world_object.material.depth_test)
        image = np.asarray(viewer.canvas.draw())
        self.assertEqual(image.shape, (480, 640, 4))
        assert_allclose(image[0, 0], [255, 255, 255, 255])
        nonwhite = np.any(image[:, :, :3] < 240, axis=2)
        self.assertGreater(nonwhite.sum(), 100)
        self.assertLess(nonwhite.mean(), 0.2)
        viewer.close()
        viewer.close()
        self.assertTrue(viewer.canvas.get_closed())

    def test_notebook_canvas_closes_twice_without_recursive_notification(self):
        async def close_canvas():
            canvas = self.NotebookCanvas(size=(100, 100))
            canvas.close()
            canvas.close()
            self.assertTrue(canvas.get_closed())
            await asyncio.sleep(0)

        asyncio.run(close_canvas())


if __name__ == '__main__':
    unittest.main()
