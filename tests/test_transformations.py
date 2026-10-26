import importlib
import subprocess
import sys
import unittest

import numpy as np
import quantities as q
from numpy.testing import assert_allclose

from coordsys import Rotation, Transformation, gpu_available


def rodrigues(axis, angle, vector):
    axis = axis / np.linalg.norm(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    )


class RotationTests(unittest.TestCase):
    def test_right_hand_rule_and_units(self):
        r = Rotation.from_axis_angle([0, 0, 100], 90 * q.deg)
        assert_allclose(r.apply([1, 0, 0]), [0, 1, 0], atol=1e-15)
        velocities = np.array([[1, 0, 0], [0, 1, 0]]) * q.m / q.s
        result = r.apply(velocities)
        self.assertEqual(result.dimensionality, velocities.dimensionality)
        assert_allclose(result.magnitude, [[0, 1, 0], [-1, 0, 0]], atol=1e-15)

    def test_random_rotations_match_rodrigues_and_invert(self):
        rng = np.random.default_rng(42)
        for _ in range(25):
            axis, vector = rng.normal(size=(2, 3))
            angle = rng.uniform(-np.pi, np.pi)
            r = Rotation.from_axis_angle(axis, angle)
            assert_allclose(r.matrix.T @ r.matrix, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(r.matrix), 1)
            assert_allclose(r.apply(vector), rodrigues(axis, angle, vector), atol=1e-14)
            assert_allclose(r.inverse().apply(r.apply(vector)), vector, atol=1e-14)

    def test_composition_order(self):
        a = Rotation.from_axis_angle([0, 0, 1], 0.4)
        b = Rotation.from_axis_angle([1, 0, 0], 1.1)
        vector = np.array([0.3, -2, 1])
        assert_allclose((a @ b).apply(vector), a.apply(b.apply(vector)), atol=1e-14)
        assert_allclose((a @ b).matrix, a.matrix @ b.matrix, atol=1e-14)

    def test_quaternion_roundtrip(self):
        rng = np.random.default_rng(7)
        for quaternion in [[1, 0, 0, 0], [0, 1, 0, 0], *rng.normal(size=(20, 4))]:
            quaternion = np.asarray(quaternion, dtype=float)
            quaternion = quaternion / np.linalg.norm(quaternion) * np.sign(quaternion[0] or 1)
            r = Rotation.from_quaternion(2 * quaternion)
            assert_allclose(r.quaternion, quaternion, atol=1e-14)
            assert_allclose(Rotation(r.matrix).matrix, r.matrix)
        half = np.sqrt(0.5)
        assert_allclose(
            Rotation.from_axis_angle([0, 0, 1], np.pi / 2).quaternion, [half, 0, 0, half]
        )

    def test_matrix_is_copied(self):
        r = Rotation()
        r.matrix[:] = 0
        assert_allclose(r.matrix, np.eye(3))

    def test_bad_input(self):
        for axis in ([0, 0, 0], [1, 2], [np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                Rotation.from_axis_angle(axis, 1)
        for angle in (np.nan, 1 * q.s, [1, 2]):
            with self.assertRaises(ValueError):
                Rotation.from_axis_angle([1, 0, 0], angle)
        for quaternion in ([0, 0, 0, 0], [1, 0, 0]):
            with self.assertRaises(ValueError):
                Rotation.from_quaternion(quaternion)
        for matrix in (2 * np.eye(3), -np.eye(3), np.eye(4)):
            with self.assertRaises(ValueError):
                Rotation(matrix)


class TransformationTests(unittest.TestCase):
    def test_identity_handles_single_batch_empty_and_nested_points(self):
        t = Transformation()
        for points in ([1, 2, 3], np.zeros((0, 3)), np.arange(18).reshape(2, 3, 3)):
            assert_allclose(t.apply(points), points)

    def test_matmul_applies_right_operand_first(self):
        move = Transformation.translation([1, 2, 3])
        turn = Transformation.rotation([0, 0, 1], np.pi / 2)
        point = [2, -3, 1]
        assert_allclose((turn @ move).apply(point), turn.apply(move.apply(point)))
        assert_allclose((turn @ move).apply([0, 0, 0]), [-2, 1, 3], atol=1e-14)

    def test_known_affine_result_and_inverse(self):
        t = (
            Transformation.scaling([2, 3, 4])
            @ Transformation.rotation([0, 0, 1], np.pi / 2)
            @ Transformation.translation([1, -2, 0.5])
        )
        points = np.array([[1, 2, 3], [-1, 0, 4]])
        assert_allclose(t.apply(points), [[0, 6, 14], [4, 0, 18]], atol=1e-14)
        assert_allclose(t.inverse().apply(t.apply(points)), points, atol=1e-14)
        assert_allclose((t.inverse() @ t).matrix, np.eye(4), atol=1e-14)

    def test_immutability_and_no_aliasing(self):
        matrix = np.eye(4)
        t = Transformation(matrix)
        matrix[0, 0] = 3
        t.matrix[1, 1] = 4
        points = np.array([[1.0, 2.0, 3.0]])
        t.apply(points)[:] = 0
        assert_allclose(t.matrix, np.eye(4))
        assert_allclose(points, [[1, 2, 3]])

    def test_units_in_points_offsets_and_composition(self):
        t = Transformation.translation([1 * q.m, 100 * q.cm, 0])
        points = np.array([[100, 0, 0]]) * q.cm
        result = t.apply(points)
        self.assertEqual(result.units, q.cm)
        assert_allclose(result.magnitude, [[200, 100, 0]])
        assert_allclose(t.inverse().apply(result).magnitude, points.magnitude)

        centimetres = Transformation.translation([100, 0, 0], unit=q.cm)
        assert_allclose(centimetres.apply([100, 0, 0]), [200, 0, 0])
        assert_allclose(centimetres.as_unit(q.m).matrix[:3, 3], [1, 0, 0])
        metres = Transformation.translation([1, 0, 0])
        assert_allclose((centimetres @ metres).apply([0, 0, 0]), [200, 0, 0])
        assert_allclose((metres @ centimetres).apply([0, 0, 0]), [2, 0, 0])

    def test_mixed_quantity_lists_and_wrong_dimensions(self):
        t = Transformation.translation([1, 0, 0], unit=q.cm)
        assert_allclose(t.apply([1 * q.m, 1 * q.cm, 1 * q.mm]), [101, 1, 0.1])
        with self.assertRaises(ValueError):
            t.apply([1 * q.s, 0, 0])
        with self.assertRaises(ValueError):
            Transformation.scaling([1 * q.m, 1, 1])
        with self.assertRaises(ValueError):
            Transformation.translation([1 * q.s, 0, 0])
        with self.assertRaises(ValueError):
            Transformation(unit=q.s)

    def test_vectors_ignore_translation_and_keep_units(self):
        t = Transformation.rotation([0, 0, 1], 90 * q.deg) @ Transformation.translation([1, 2, 3])
        result = t.apply(np.array([1, 0, 0]) * q.m / q.s, vectors=True)
        self.assertEqual(result.units, q.m / q.s)
        assert_allclose(result.magnitude, [0, 1, 0], atol=1e-15)

    def test_float32_points_stay_float32(self):
        t = Transformation.rotation([1, 2, 3], 0.3)
        points = np.ones((4, 3), dtype=np.float32)
        self.assertEqual(t.apply(points).dtype, np.float32)
        self.assertEqual(t.apply([1, 2, 3]).dtype, np.float64)

    def test_float64_precision_survives_inverse_and_composition(self):
        t = Transformation.translation([100000000.25, 0, 0])
        self.assertEqual((t @ Transformation()).matrix[0, 3], 100000000.25)
        self.assertEqual(t.inverse().matrix[0, 3], -100000000.25)
        assert_allclose(t.inverse().apply(t.apply([0.125, 0, 0])), [0.125, 0, 0], atol=0, rtol=0)

    def test_scaling_accepts_one_or_three_factors(self):
        assert_allclose(Transformation.scaling(2).apply([1, 2, 3]), [2, 4, 6])
        assert_allclose(Transformation.scaling([1, 2, 3]).apply([1, 1, 1]), [1, 2, 3])

    def test_all_shear_planes(self):
        for plane in ('xy', 'xz', 'yx', 'yz', 'zx', 'zy'):
            expected = np.array([1.0, 2.0, 3.0])
            expected['xyz'.index(plane[0])] += expected['xyz'.index(plane[1])]
            assert_allclose(Transformation.shear(plane, 45 * q.deg).apply([1, 2, 3]), expected)
        with self.assertRaises(ValueError):
            Transformation.shear('xy', 90 * q.deg)
        with self.assertRaises(ValueError):
            Transformation.shear('xx', 0.1)

    def test_look_at(self):
        eye = np.array([1, 2, 3]) * q.m
        target = np.array([100, 200, 0]) * q.cm
        t = Transformation.look_at(eye, target)
        assert_allclose(t.apply(eye).magnitude, [0, 0, 0], atol=1e-15)
        assert_allclose(t.apply(target).rescale(q.m).magnitude, [0, 0, -3])
        for args in [([0, 0, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, 0], [0, 0, 2])]:
            with self.assertRaises(ValueError):
                Transformation.look_at(*args)

    def test_bad_matrices_and_points(self):
        for points in (1, [1, 2], np.ones((3, 4))):
            with self.assertRaises(ValueError):
                Transformation().apply(points)
        projective = np.eye(4)
        projective[3, 2] = 0.5
        for matrix in (np.eye(3), np.full((4, 4), np.nan), projective):
            with self.assertRaises(ValueError):
                Transformation(matrix)
        with self.assertRaises(np.linalg.LinAlgError):
            Transformation.scaling([0, 1, 1]).inverse()


class BackendTests(unittest.TestCase):
    def test_import_needs_neither_gpu_nor_plotting(self):
        subprocess.run(
            [
                sys.executable,
                '-B',
                '-c',
                "import coordsys, sys; assert not {'cupy','pyvista','matplotlib'} & sys.modules.keys()",
            ],
            check=True,
        )


@unittest.skipUnless(gpu_available(), 'working CuPy/CUDA device unavailable')
class GPUTests(unittest.TestCase):
    def test_gpu_points_stay_on_gpu(self):
        cp = importlib.import_module('cupy')
        t = Transformation.translation([1, 2, 3]) @ Transformation.rotation([1, 2, 3], 0.4)
        points = np.arange(12, dtype=np.float64).reshape(4, 3)
        result = t.apply(cp.asarray(points))
        self.assertIsInstance(result, cp.ndarray)
        assert_allclose(cp.asnumpy(result), t.apply(points), atol=1e-13)

    def test_gpu_frame_points_stay_on_gpu(self):
        cp = importlib.import_module('cupy')
        from coordsys import CoordinateSystem

        root = CoordinateSystem()
        child = CoordinateSystem(parent=root)
        child.add_points(cp.array([[1.0, 0.0, 0.0]]), 'p')
        root.translate([1, 0, 0])
        world = child.get_points('p', space='world')
        self.assertIsInstance(world, cp.ndarray)
        assert_allclose(cp.asnumpy(world), [[2, 0, 0]])
        child.rotate([0, 0, 1], 0.7, inherit=False)
        self.assertIsInstance(child.get_points('p'), cp.ndarray)
        assert_allclose(cp.asnumpy(child.get_points('p', space='world')), [[2, 0, 0]], atol=1e-13)


if __name__ == '__main__':
    unittest.main()
