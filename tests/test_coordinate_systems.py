import unittest

import numpy as np
import quantities as q
from numpy.testing import assert_allclose

from coordsys import CoordinateSystem


def metres(position):
    return position.rescale(q.m).magnitude


class FrameTests(unittest.TestCase):
    def test_local_world_roundtrip_translated_rotated_scaled_skew_frame(self):
        frame = CoordinateSystem(u=[2, 0, 0], v=[1, 3, 0], w=[0, 0, 4], origin=[1, 2, 3])
        frame.rotate_euler_local([1, 1, 0], 0.7)
        points = np.random.default_rng(5).normal(size=(2, 8, 3)) * q.cm
        world = frame.to_global(points)
        self.assertEqual(world.units, q.cm)
        assert_allclose(frame.to_local(world).magnitude, points.magnitude, atol=1e-12)
        assert_allclose(
            frame.get_local_coordinates(frame.get_global_coordinates([1, 2, 3])), [1, 2, 3]
        )

    def test_known_world_conversion_uses_basis_and_origin(self):
        frame = CoordinateSystem(origin=[1, 0, 0])
        frame.rotate_euler_local([0, 0, 1], 90 * q.deg)
        assert_allclose(frame.to_global([1, 0, 0]), [1, 1, 0], atol=1e-14)
        assert_allclose(frame.to_local([1, 1, 0]), [1, 0, 0], atol=1e-14)

    def test_grandchild_composes_all_ancestors(self):
        root = CoordinateSystem(origin=[1, 0, 0])
        root.rotate_euler_local([0, 0, 1], 90 * q.deg)
        child = root.add_child(CoordinateSystem(origin=[1, 0, 0]), keep_world=False)
        grandchild = child.add_child(CoordinateSystem(origin=[0, 1, 0]), keep_world=False)
        assert_allclose(grandchild.to_global([1, 0, 0]), [0, 2, 0], atol=1e-14)
        assert_allclose(grandchild.to_local([0, 2, 0]), [1, 0, 0], atol=1e-14)

    def test_deep_hierarchy_has_no_recursion_limit(self):
        root = CoordinateSystem(origin=[1, 2, 3])
        leaf = root
        for _ in range(1100):
            leaf = leaf.add_child()
        root.translate([1, 0, 0])
        assert_allclose(leaf.to_global([1, 0, 0]), [3, 2, 3])

    def test_new_child_inherits_parent_pose_unit_dtype_and_backend(self):
        root = CoordinateSystem(origin=[100, 0, 0], unit=q.cm, dtype=np.float32)
        root.rotate_euler_local([0, 1, 0], 0.5)
        child = root.add_child()
        assert_allclose(child.world_transform.matrix, root.world_transform.matrix)
        assert_allclose(child.local_transform.matrix, np.eye(4))
        self.assertEqual(child.dtype, np.float32)
        self.assertEqual(child.unit, q.cm)

    def test_local_rotation_about_translated_origin(self):
        frame = CoordinateSystem(origin=[1, 0, 0])
        frame.add_points([[2, 0, 0]], 'p')
        frame.rotate_euler_local([0, 0, 1], 90 * q.deg)
        assert_allclose(frame.get_points('p'), [[1, 1, 0]], atol=1e-14)
        assert_allclose(metres(frame.origin), [1, 0, 0])

    def test_world_pivot_moves_frame_points_and_descendants_together(self):
        root = CoordinateSystem(origin=[2, 0, 0])
        root.add_points([[3, 0, 0]], 'p')
        child = root.add_child(CoordinateSystem(origin=[4, 0, 0]))
        child.add_points([[5, 0, 0]], 'p')
        grandchild = child.add_child(CoordinateSystem(origin=[6, 0, 0]))
        root.rotate_euler([1, 0, 0], [0, 0, 1], 90 * q.deg)
        assert_allclose(metres(root.origin), [1, 1, 0], atol=1e-14)
        assert_allclose(root.get_points('p'), [[1, 2, 0]], atol=1e-14)
        assert_allclose(metres(child.origin), [1, 3, 0], atol=1e-14)
        assert_allclose(child.get_points('p'), [[1, 4, 0]], atol=1e-14)
        assert_allclose(metres(grandchild.origin), [1, 5, 0], atol=1e-14)

    def test_mixed_unit_translation_updates_origin_and_geometry_consistently(self):
        frame = CoordinateSystem()
        frame.add_points(np.array([[100, 0, 0]]) * q.cm, 'p')
        frame.translate(np.array([1, 0, 0]) * q.m)
        assert_allclose(frame.get_points('p').magnitude, [[200, 0, 0]])
        assert_allclose(metres(frame.origin), [1, 0, 0])
        frame.set_cartesian_coordinates(1 * q.m, 100 * q.cm, 1000 * q.mm)
        assert_allclose(metres(frame.origin), [1, 1, 1])

    def test_parent_and_child_different_units(self):
        root = CoordinateSystem(origin=[100, 0, 0], unit=q.cm)
        child = root.add_child(CoordinateSystem(origin=[1000, 0, 0], unit=q.mm), keep_world=False)
        child.add_points([[1000, 0, 0]], 'p', space='local')
        root.translate([100, 0, 0])
        assert_allclose(metres(child.origin), [3, 0, 0])
        assert_allclose(child.get_points('p'), [[4000, 0, 0]])
        assert_allclose(child.to_local([4000, 0, 0]), [1000, 0, 0])

    def test_attached_point_lists_rescale_mixed_units(self):
        frame = CoordinateSystem(unit=q.cm)
        frame.add_points([[1 * q.m, 1 * q.cm, 1 * q.mm]], 'p')
        assert_allclose(frame.get_points('p'), [[100, 1, 0.1]])
        with self.assertRaises(ValueError):
            frame.add_points([[1 * q.s, 0, 0]])

    def test_scaling_preserves_rotation_and_scales_child_offsets(self):
        root = CoordinateSystem(origin=[1, 0, 0])
        root.rotate_euler_local([0, 0, 1], 90 * q.deg)
        child = root.add_child(CoordinateSystem(origin=[1, 1, 0]))
        root.scale(2)
        assert_allclose(root.u, [0, 2, 0], atol=1e-14)
        assert_allclose(metres(child.origin), [1, 2, 0], atol=1e-14)
        root.scale(3)
        assert_allclose(root.u, [0, 6, 0], atol=1e-14)
        assert_allclose(metres(child.origin), [1, 6, 0], atol=1e-14)

    def test_nonuniform_scale_is_along_current_axes(self):
        frame = CoordinateSystem()
        frame.rotate_euler_local([0, 0, 1], 90 * q.deg)
        frame.add_points([[1, 1, 0]], 'p', space='local')
        frame.scale_pointwise(2, 3, 4)
        assert_allclose(frame.get_points('p'), [[-3, 2, 0]], atol=1e-14)
        frame.normalize()
        assert_allclose(np.linalg.norm(frame.world_transform.matrix[:3, :3], axis=0), [1, 1, 1])

    def test_inherit_false_preserves_attached_world_geometry_for_all_pose_changes(self):
        operations = [
            lambda c: c.translate([1, 2, 3], inherit=False),
            lambda c: c.rotate_euler([3, 2, 1], [0, 1, 0], 0.5, inherit=False),
            lambda c: c.scale_pointwise(2, 3, 4, inherit=False),
            lambda c: c.set_cartesian_coordinates(5, 6, 7, inherit=False),
            lambda c: c.set_spherical_coordinates(3, 0.4, 0.6, inherit=False),
            lambda c: c.normalize(inherit=False),
            lambda c: c.to_opposite(inherit=False),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                root = CoordinateSystem(origin=[1, 2, 3])
                root.scale(2)
                root.add_points(np.array([[2, 3, 4]]) * q.cm, 'p')
                child = root.add_child(CoordinateSystem(origin=[2, 0, 0]))
                child.add_points([[3, 2, 1]], 'p')
                leaf = child.add_child(CoordinateSystem(origin=[4, 5, 6]))
                before = [
                    root.get_points('p'),
                    child.world_transform.matrix,
                    child.get_points('p'),
                    leaf.world_transform.matrix,
                ]
                operation(root)
                after = [
                    root.get_points('p'),
                    child.world_transform.matrix,
                    child.get_points('p'),
                    leaf.world_transform.matrix,
                ]
                for actual, expected in zip(after, before):
                    if isinstance(expected, q.Quantity):
                        self.assertEqual(actual.units, expected.units)
                        actual = actual.rescale(expected.units).magnitude
                        expected = expected.magnitude
                    assert_allclose(actual, expected, atol=1e-10)

    def test_cartesian_setter_translates_children_without_collapsing_offsets(self):
        root = CoordinateSystem()
        child = root.add_child(CoordinateSystem(origin=[1, 0, 0]))
        root.set_cartesian_coordinates(10, 0, 0)
        assert_allclose(metres(child.origin), [11, 0, 0])

    def test_spherical_setter_rotates_attached_geometry_and_offsets(self):
        root = CoordinateSystem()
        root.add_points([[1, 0, 0]], 'p', space='local')
        child = root.add_child(CoordinateSystem(origin=[1, 0, 0]))
        root.set_spherical_coordinates(200 * q.cm, 90 * q.deg, 90 * q.deg)
        assert_allclose(metres(root.origin), [0, 2, 0], atol=1e-14)
        assert_allclose(root.u, [0, 0, -1], atol=1e-14)
        assert_allclose(root.w, [0, 1, 0], atol=1e-14)
        assert_allclose(root.get_points('p'), [[0, 2, -1]], atol=1e-14)
        assert_allclose(metres(child.origin), [0, 2, -1], atol=1e-14)

    def test_labelled_local_rotation_uses_child_origin(self):
        root = CoordinateSystem()
        child = root.add_child(CoordinateSystem(origin=[1, 0, 0]), 'c')
        child.add_points([[2, 0, 0]], 'p')
        root.rotate_euler_local([0, 0, 1], 90 * q.deg, label='c')
        assert_allclose(child.get_points('p'), [[1, 1, 0]], atol=1e-14)
        with self.assertRaises(KeyError):
            root.set_spherical_coordinates(1, 0, 0, label='missing')

    def test_reparent_detach_and_cycle_protection(self):
        a, b = CoordinateSystem(origin=[1, 2, 3]), CoordinateSystem(origin=[4, 5, 6])
        child = a.add_child()
        child.add_points([[1, 0, 0]], 'p', space='local')
        before = child.world_transform.matrix, child.get_points('p')
        b.add_child(child, 'moved')
        self.assertFalse(a.children)
        self.assertIs(child.get_parent(), b)
        assert_allclose(child.world_transform.matrix, before[0])
        assert_allclose(child.get_points('p'), before[1])
        with self.assertRaises(ValueError):
            child.add_child(b)
        with self.assertRaises(ValueError):
            a.add_child(a)
        self.assertIs(b.remove_child('moved'), child)
        self.assertIsNone(child.get_parent())
        assert_allclose(child.world_transform.matrix, before[0])
        with self.assertRaises(TypeError):
            a.children['bad'] = b

    def test_symmetry_and_opposite_use_frame_local_offsets(self):
        root = CoordinateSystem(origin=[1, 2, 0])
        root.rotate_euler_local([0, 0, 1], 90 * q.deg)
        symmetric = root.get_symmetric(100 * q.cm, axis=[1, 0, 0])
        assert_allclose(metres(symmetric.origin), [1, 3, 0], atol=1e-14)
        assert_allclose(symmetric.u, root.u)
        antisymmetric = root.get_antisymmetric(1, axis=[1, 0, 0])
        assert_allclose(antisymmetric.u, -root.u)
        root.to_opposite().to_opposite()
        assert_allclose(root.u, symmetric.u)
        root.add_symmetric_child(label='same')
        root.add_antisymmetric_child(label='opposite')

    def test_point_storage_labels_no_aliasing_and_local_world_modes(self):
        frame = CoordinateSystem(origin=[1, 0, 0])
        data = np.array([[2.0, 0.0, 0.0]])
        label = frame.add_points(data)
        data[:] = 99
        assert_allclose(frame.get_points(label), [[2, 0, 0]])
        assert_allclose(frame.get_points(label, space='local'), [[1, 0, 0]])
        frame.get_points(label)[:] = 0
        assert_allclose(frame.get_points(label), [[2, 0, 0]])
        for _ in range(30):
            frame.add_points([[0, 0, 0]])
        before = frame.get_points('C')
        self.assertTrue(frame.remove_points('A'))
        self.assertFalse(frame.remove_points('A'))
        frame.add_points([[7, 8, 9]])
        assert_allclose(frame.get_points('C'), before)
        with self.assertRaises(ValueError):
            frame.add_points([[0, 0, 0]], 'C')
        with self.assertRaises(ValueError):
            frame.add_points([[0, 0, 0]], space='typo')
        with self.assertRaises(ValueError):
            frame.get_points(space='typo')

    def test_pose_accessors_are_snapshots_and_setters_move_attachments(self):
        frame = CoordinateSystem()
        frame.add_points([[1, 0, 0]], 'p')
        snapshot = frame.world_transform
        frame.u[:] = 99
        frame.origin[:] = 99 * q.m
        assert_allclose(frame.to_global([1, 0, 0]), [1, 0, 0])
        frame.origin = [1, 0, 0]
        frame.u = [2, 0, 0]
        assert_allclose(frame.get_points('p'), [[3, 0, 0]])
        assert_allclose(snapshot.matrix, np.eye(4))

    def test_invalid_pose_changes_are_atomic(self):
        frame = CoordinateSystem(origin=[1, 2, 3])
        before = frame.world_transform.matrix
        with self.assertRaises(ValueError):
            frame.scale(0)
        with self.assertRaises(ValueError):
            frame.v = frame.u
        with self.assertRaises(ValueError):
            frame.set_spherical_coordinates(-1, 0, 0)
        with self.assertRaises(ValueError):
            frame.translate([np.inf, 0, 0])
        assert_allclose(frame.world_transform.matrix, before)
        with self.assertRaises(ValueError):
            CoordinateSystem(u=[1, 0, 0], v=[1, 0, 0])

    def test_axis_translation_normalizes_direction(self):
        frame = CoordinateSystem()
        frame.translate_along_axis([2, 0, 0], 100 * q.cm)
        assert_allclose(metres(frame.origin), [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
