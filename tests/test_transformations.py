import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import quantities as q
from numpy.testing import assert_allclose

from coordsys import Backend, CoordinateSystem, Quaternion, RotationQuaternion, Transformation


class QuaternionTests(unittest.TestCase):
    def test_hamilton_product_is_noncommutative(self):
        i, j = Quaternion(0, 1, 0, 0), Quaternion(0, 0, 1, 0)
        assert_allclose((i * j).q, [0, 0, 0, 1])
        assert_allclose((j * i).q, [0, 0, 0, -1])

    def test_nonunit_inverse_both_orders(self):
        for components in [(2, 0, 0, 0), (1, 2, -3, 4)]:
            a = Quaternion(*components)
            assert_allclose((a * a.inverse()).q, [1, 0, 0, 0], atol=1e-14)
            assert_allclose((a.inverse() * a).q, [1, 0, 0, 0], atol=1e-14)

    def test_inverse_is_not_stale_after_mutation(self):
        a = Quaternion(2, 0, 0, 0)
        a.inverse()
        a.normalize()
        assert_allclose(a.inverse().q, [1, 0, 0, 0])
        a.q *= 3
        assert_allclose((a * a.inverse()).q, [1, 0, 0, 0])

    def test_zero_and_invalid_quaternions_rejected(self):
        for operation in ('inverse', 'normalize'):
            with self.assertRaises(ValueError):
                getattr(Quaternion(0, 0, 0, 0), operation)()
        with self.assertRaises(ValueError):
            Quaternion(np.nan, 0, 0, 0)
        with self.assertRaises(ValueError):
            Quaternion(1, 0, 0, 0) / 0

    def test_algebra_helpers(self):
        a, b = Quaternion(0, 1, 2, 3), Quaternion(0, 4, 5, 6)
        assert_allclose(Quaternion.cross_product(a, b).q, [0, -3, 6, -3])
        assert_allclose(Quaternion.dot_product(a, b).q, [32, 0, 0, 0])
        assert_allclose(Quaternion.scalar_product(a, 2).q, [0, 2, 4, 6])
        assert_allclose(((a + b) - b).q, a.q)

    def test_axis_angle_right_hand_rule_and_units(self):
        r = RotationQuaternion(90 * q.deg, [0, 0, 100])
        assert_allclose(r.rotate([1, 0, 0]), [0, 1, 0], atol=1e-14)
        vectors = np.array([[1, 0, 0], [0, 1, 0]]) * q.m / q.s
        result = r.rotate(vectors)
        self.assertEqual(result.dimensionality, vectors.dimensionality)
        assert_allclose(result.magnitude, [[0, 1, 0], [-1, 0, 0]], atol=1e-14)
        assert_allclose(r.matrix, RotationQuaternion(np.pi / 2, [0, 0, 1]).matrix)

    def test_random_rotations_preserve_lengths_orientation_and_inverse(self):
        rng = np.random.default_rng(42)
        for _ in range(25):
            axis, vector = rng.normal(size=(2, 3))
            angle = rng.uniform(-np.pi, np.pi)
            r = RotationQuaternion(angle, axis)
            matrix = r.matrix[:3, :3]
            assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(matrix), 1)
            assert_allclose(np.linalg.norm(r.rotate(vector)), np.linalg.norm(vector), atol=1e-14)
            assert_allclose(
                RotationQuaternion(-angle, axis).rotate(r.rotate(vector)), vector, atol=1e-14
            )
            # Rodrigues' formula is an independent reference for the quaternion.
            axis /= np.linalg.norm(axis)
            expected = (
                vector * np.cos(angle)
                + np.cross(axis, vector) * np.sin(angle)
                + axis * np.dot(axis, vector) * (1 - np.cos(angle))
            )
            assert_allclose(r.rotate(vector), expected, atol=1e-14)

    def test_bad_rotation_input(self):
        for axis in ([0, 0, 0], [1, 2], [np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                RotationQuaternion(1, axis)
        with self.assertRaises(ValueError):
            RotationQuaternion(np.nan, [1, 0, 0])
        with self.assertRaises(ValueError):
            RotationQuaternion(1 * q.s, [1, 0, 0])

    def test_dtype_preserved(self):
        for dtype, atol in [(np.float32, 2e-6), (np.float64, 1e-14)]:
            r = RotationQuaternion(np.pi / 7, [1, 2, 3], dtype=dtype)
            self.assertEqual(r.matrix.dtype, dtype)
            self.assertEqual(r.inverse().dtype, dtype)
            assert_allclose((r * r.inverse()).q, [1, 0, 0, 0], atol=atol)


class TransformationTests(unittest.TestCase):
    def test_identity_single_batch_empty_and_leading_dimensions(self):
        t = Transformation()
        for points in ([1, 2, 3], np.zeros((0, 3)), np.arange(18).reshape(2, 3, 3)):
            assert_allclose(t.apply(points), points)
        view = np.arange(30).reshape(10, 3)[::2]
        assert_allclose(t.apply(view), view)

    def test_known_affine_operations_and_inverse(self):
        t = Transformation().translation(1, -2, 0.5).rotation_z(np.pi / 2).scaling(2, 3, 4)
        points = np.array([[1, 2, 3], [-1, 0, 4]])
        assert_allclose(t.apply(points), [[0, 6, 14], [4, 0, 18]], atol=1e-14)
        assert_allclose(t.inverse().apply(t.apply(points)), points, atol=1e-14)
        assert_allclose(t.apply(points[0]), t.apply(points)[0])

    def test_compose_order_and_independence(self):
        a = Transformation().translation(1, 2, 3)
        b = Transformation().rotation_z(np.pi / 2)
        c = a.compose(b)
        assert_allclose(c.apply([2, -3, 1]), b.apply(a.apply([2, -3, 1])))
        a.reset()
        assert_allclose(c.apply([0, 0, 0]), [-2, 1, 3], atol=1e-14)

    def test_mixed_length_units(self):
        t = Transformation().translation(1 * q.m, 100 * q.cm, 0)
        points = np.array([[100, 0, 0]]) * q.cm
        result = t.apply(points)
        assert_allclose(result.magnitude, [[200, 100, 0]])
        self.assertEqual(result.units, q.cm)
        assert_allclose(t.inverse().apply(result).magnitude, points.magnitude)

    def test_custom_unit_numeric_input_and_quantity_vector(self):
        t = Transformation(unit=q.cm).from_vector(np.array([1, 0, 0]) * q.m)
        assert_allclose(t.apply([100, 0, 0]), [200, 0, 0])
        assert_allclose(t.apply(np.array([1, 0, 0]) * q.m).magnitude, [2, 0, 0])
        assert_allclose(t.as_unit(q.m).matrix[:3, 3], [1, 0, 0])

    def test_composition_converts_transform_units(self):
        a = Transformation(unit=q.cm).translation(100, 0, 0)
        b = Transformation(unit=q.m).translation(1, 0, 0)
        assert_allclose(a.compose(b).apply([0, 0, 0]), [200, 0, 0])
        assert_allclose(b.compose(a).apply([0, 0, 0]), [2, 0, 0])

    def test_mixed_quantity_lists_do_not_silently_drop_units(self):
        t = Transformation(unit=q.cm).translation(1 * q.m)
        assert_allclose(t.apply([1 * q.m, 1 * q.cm, 1 * q.mm]), [200, 1, 0.1])
        with self.assertRaises(ValueError):
            t.apply([1 * q.s, 0, 0])
        with self.assertRaises(ValueError):
            t.scaling(1 * q.m, 1, 1)

    def test_float64_precision_survives_inverse_composition_and_rotation(self):
        t = Transformation(dtype=np.float64).translation(100000000.25)
        self.assertEqual(t.inverse().matrix.dtype, np.float64)
        self.assertEqual(t.compose(Transformation()).matrix[0, 3], 100000000.25)
        self.assertEqual(t.inverse().matrix[0, 3], -100000000.25)
        assert_allclose(t.inverse().apply(t.apply([0.125, 0, 0])), [0.125, 0, 0], atol=0, rtol=0)
        r = Transformation(dtype=np.float64).rotation(0.123456789012345, [1, 0, 0])
        self.assertAlmostEqual(r.matrix[1, 1], np.cos(0.123456789012345), places=14)

    def test_float32_operations_stay_float32_and_mixed_composition_promotes(self):
        t = (
            Transformation(dtype=np.float32)
            .rotation(0.3, [1, 2, 3])
            .translation(1)
            .scaling(2, 3, 4)
        )
        self.assertEqual(t.matrix.dtype, np.float32)
        self.assertEqual(t.apply([1, 2, 3]).dtype, np.float32)
        self.assertEqual(t.inverse().dtype, np.float32)
        self.assertEqual(t.compose(Transformation(dtype=np.float64)).dtype, np.float64)

    def test_mixed_dtype_composition_promotes_before_unit_conversion(self):
        a = Transformation(dtype=np.float64, unit=q.m)
        b = Transformation(dtype=np.float32, unit=q.cm).translation(3)
        assert_allclose(a.compose(b).apply([0, 0, 0]), [0.03, 0, 0], rtol=0, atol=1e-16)

    def test_direction_vectors_ignore_translation_and_retain_velocity_units(self):
        t = Transformation().translation(100, 200, 300).rotation_z(90 * q.deg)
        result = t.apply(np.array([1, 0, 0]) * q.m / q.s, vectors=True)
        assert_allclose(result.magnitude, [0, 1, 0], atol=1e-14)
        self.assertEqual(result.units, q.m / q.s)

    def test_all_shears(self):
        for name, row, column in [
            ('xy', 0, 1),
            ('xz', 0, 2),
            ('yx', 1, 0),
            ('yz', 1, 2),
            ('zx', 2, 0),
            ('zy', 2, 1),
        ]:
            t = getattr(Transformation(), 'shear_' + name)(45 * q.deg)
            expected = np.array([1.0, 2.0, 3.0])
            expected[row] += expected[column]
            assert_allclose(t.apply([1, 2, 3]), expected)
        with self.assertRaises(ValueError):
            Transformation().shear_xy(90 * q.deg)

    def test_look_at_and_degenerate_input(self):
        eye = np.array([1, 2, 3]) * q.m
        target = np.array([100, 200, 0]) * q.cm
        t = Transformation().look_at(eye, target)
        assert_allclose(t.apply(eye).magnitude, [0, 0, 0])
        assert_allclose(t.apply(target).rescale(q.m).magnitude, [0, 0, -3])
        for args in [([0, 0, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, 0], [0, 0, 2])]:
            with self.assertRaises(ValueError):
                Transformation().look_at(*args)

    def test_projective_single_and_batch_agree_without_arbitrary_cutoff(self):
        matrix = np.eye(4)
        matrix[3] = [0, 0, 0, 1e-12]
        t = Transformation(matrix)
        expected = np.array([1, 2, 3]) * 1e12
        assert_allclose(t.apply([1, 2, 3]), expected)
        assert_allclose(t.apply([[1, 2, 3]])[0], expected)
        with self.assertRaises(ValueError):
            t.apply([1, 2, 3], vectors=True)
        matrix[3] = [0, 0, 1, 1]
        for point in ([1, 2, -1], [[1, 2, -1]]):
            with self.assertRaises(ValueError):
                Transformation(matrix).apply(point)

    def test_projective_unit_change_preserves_physical_map(self):
        matrix = np.eye(4)
        matrix[3, 2] = 0.5
        t = Transformation(matrix)
        p = np.array([1, 2, 3]) * q.m
        assert_allclose(t.apply(p).magnitude, t.as_unit(q.cm).apply(p).magnitude)

    def test_bad_shapes_units_and_singular_inverse(self):
        for points in (1, [1, 2], np.ones((3, 4))):
            with self.assertRaises(ValueError):
                Transformation().apply(points)
        for matrix in (np.eye(3), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                Transformation(matrix)
        with self.assertRaises(ValueError):
            Transformation(dtype=int)
        with self.assertRaises(ValueError):
            Transformation(unit=q.s)
        with self.assertRaises(ValueError):
            Transformation().translation(1 * q.s)
        with self.assertRaises(np.linalg.LinAlgError):
            Transformation().scaling(0, 1, 1).inverse()

    def test_matrix_and_input_do_not_alias(self):
        matrix = np.eye(4)
        t = Transformation(matrix)
        matrix[0, 0] = 3
        exposed = t.matrix
        exposed[1, 1] = 4
        points = np.array([[1.0, 2.0, 3.0]])
        result = t.apply(points)
        result[:] = 0
        assert_allclose(t.matrix, np.eye(4))
        assert_allclose(points, [[1, 2, 3]])


class BackendTests(unittest.TestCase):
    def test_cpu_import_needs_neither_gpu_nor_plotting(self):
        subprocess.run(
            [
                sys.executable,
                '-B',
                '-c',
                "import coordsys, sys; assert not {'cupy','pyvista','matplotlib'} & sys.modules.keys()",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_default_aliases_and_invalid_devices(self):
        self.assertIs(Backend().get(), np)
        for name in ['cpu', 'numpy', 'CPU']:
            self.assertIs(Backend(name).get(), np)
        with self.assertRaises(ValueError):
            Backend('typo')

    def test_missing_cupy_is_explicit_not_a_silent_cpu_benchmark(self):
        with patch(
            'coordsys.transformations.importlib.import_module', side_effect=ImportError('missing')
        ):
            self.assertFalse(Backend.gpu_available())
            with self.assertRaisesRegex(RuntimeError, 'CuPy/CUDA'):
                Backend('gpu')
        t = Transformation()
        t.backend.use_cpu()
        assert_allclose(t.apply([1, 2, 3]), [1, 2, 3])


@unittest.skipUnless(Backend.gpu_available(), 'working CuPy/CUDA device unavailable')
class GPUTests(unittest.TestCase):
    def test_residency_cpu_boundary_backend_switch_and_dtype(self):
        cp = importlib.import_module('cupy')
        t = (
            Transformation(device='gpu', dtype=np.float64)
            .rotation(0.4, [1, 2, 3])
            .translation(1, 2, 3)
        )
        points = np.arange(12, dtype=np.float64).reshape(4, 3)
        gpu_points = cp.asarray(points)
        result = t.apply(gpu_points)
        self.assertIsInstance(result, cp.ndarray)
        expected = Transformation(t.matrix).apply(points)
        assert_allclose(cp.asnumpy(result), expected, atol=1e-13)
        self.assertIsInstance(t.apply(points), np.ndarray)
        self.assertIsInstance(t.apply(points, return_numpy=False), cp.ndarray)
        assert_allclose(t.apply(gpu_points, return_numpy=True), expected, atol=1e-13)
        t.backend.use_cpu()
        self.assertIsInstance(t.apply(gpu_points), np.ndarray)
        assert_allclose(t.apply(gpu_points), expected, atol=1e-13)

    def test_gpu_quantities_and_inverse(self):
        t = Transformation(device='gpu').translation(1 * q.m)
        points = np.array([[100, 0, 0]]) * q.cm
        assert_allclose(t.apply(points).magnitude, [[200, 0, 0]])
        assert_allclose(t.inverse().apply(t.apply(points)).magnitude, points.magnitude)

    def test_gpu_frame_geometry_remains_resident(self):
        cp = importlib.import_module('cupy')
        root = CoordinateSystem(device='gpu')
        child = root.add_child()
        child.add_points(cp.array([[1.0, 0.0, 0.0]]), 'p', space='local')
        root.translate([1, 0, 0])
        self.assertIsInstance(child.get_points('p'), cp.ndarray)
        assert_allclose(cp.asnumpy(child.get_points('p')), [[2, 0, 0]])
        child.rotate_euler_local([0, 0, 1], 0.7, inherit=False)
        self.assertIsInstance(child.get_points('p'), cp.ndarray)
        assert_allclose(cp.asnumpy(child.get_points('p')), [[2, 0, 0]], atol=1e-13)


if __name__ == '__main__':
    unittest.main()
