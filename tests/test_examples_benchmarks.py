import contextlib
import importlib.util
import io
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from examples.simple_transform import make_scene
from performance.benchmark_frames import run_benchmarks as run_frame_benchmarks
from performance.benchmark_transformations import run_benchmarks
from performance.run_benchmarks import main


class UsageTests(unittest.TestCase):
    def test_example_scene_roundtrip(self):
        root, child = make_scene()
        points = child.get_points('points', space='world')
        self.assertEqual(points.shape, (1000, 3))
        assert_allclose(child.to_world(child.to_local(points)), points, rtol=0, atol=1e-13)
        self.assertIs(child.parent, root)

    def test_readme_python_examples(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        namespace = {}
        for code in re.findall(r'```python\n(.*?)```', readme.read_text(), re.DOTALL):
            exec(compile(code, str(readme), 'exec'), namespace)

    def test_cpu_only_benchmark_never_requests_gpu(self):
        with patch(
            'coordsys.backend.gpu_available', side_effect=AssertionError('must not probe GPU')
        ):
            rows = run_benchmarks(
                [3, 10],
                num_runs=2,
                cpu_only=True,
                operations=('Combined', 'Translation', 'Rotation', 'Scaling'),
            )
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row['device'], 'cpu')
            self.assertLess(row['max_abs_error'], 1e-6)
            for key in ('setup_s', 'upload_s', 'apply_s', 'download_s'):
                self.assertGreaterEqual(row[key], 0)

    def test_missing_gpu_omitted_and_cli_operations_dispatch(self):
        with patch('coordsys.backend.gpu_available', return_value=False):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = main(['--test', 'operations', '--points', '3', '--runs', '1'])
        self.assertEqual({row['operation'] for row in rows}, {'Translation', 'Rotation', 'Scaling'})
        self.assertEqual({row['device'] for row in rows}, {'cpu'})

    def test_frame_benchmark_runs(self):
        rows = run_frame_benchmarks(depths=(1, 3), min_time=0.001)
        self.assertEqual({depth for _, depth, _ in rows}, {0, 1, 3})
        self.assertTrue(all(time > 0 for _, _, time in rows))

    def test_benchmark_rejects_invalid_run_counts(self):
        for sizes, runs in [([], 1), ([0], 1), ([2], 0)]:
            with self.assertRaises(ValueError):
                run_benchmarks(sizes, runs, cpu_only=True)


@unittest.skipUnless(
    importlib.util.find_spec('pyvista') and importlib.util.find_spec('matplotlib'),
    'optional visualization dependencies unavailable',
)
class VisualizationTests(unittest.TestCase):
    def test_world_geometry_and_flags_without_opening_a_window(self):
        import quantities as q

        from coordsys import CoordinateSystem

        class Plotter:
            def __init__(self):
                self.arrows, self.points = [], []

            def add_arrows(self, origin, axis, **kwargs):
                self.arrows.append((origin.copy(), axis.copy()))

            def add_points(self, points, **kwargs):
                self.points.append(np.array(points, copy=True))

        root = CoordinateSystem(unit=q.cm)
        child = CoordinateSystem([100, 0, 0], parent=root)
        child.add_points([[100, 0, 0]], 'p')
        plotter = Plotter()
        self.assertIs(root.visualize(plotter=plotter), plotter)
        self.assertEqual(len(plotter.arrows), 6)
        assert_allclose(plotter.points[-1], [[200, 0, 0]])
        plotter = Plotter()
        root.visualize(plotter=plotter, children=False, show_points=False)
        self.assertEqual(len(plotter.arrows), 3)
        self.assertEqual(len(plotter.points), 1)


if __name__ == '__main__':
    unittest.main()
