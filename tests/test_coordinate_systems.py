import unittest

import numpy as np
import quantities as q
from numpy.testing import assert_allclose

from coordsys import CoordinateSystem, Rotation


def metres(position):
    return position.rescale(q.m).magnitude


def turned(origin=(0, 0, 0), angle=90 * q.deg, axis=(0, 0, 1), **kwargs):
    return CoordinateSystem(origin, Rotation.from_axis_angle(axis, angle), **kwargs)


class ConversionTests(unittest.TestCase):
    def test_world_local_roundtrip(self):
        frame = CoordinateSystem([1, 2, 3], Rotation.from_axis_angle([1, 1, 0], 0.7), scale=-2.5)
        points = np.random.default_rng(5).normal(size=(2, 8, 3)) * q.cm
        world = frame.to_world(points)
        self.assertEqual(world.units, q.cm)
        assert_allclose(frame.to_local(world).magnitude, points.magnitude, atol=1e-12)

    def test_known_conversion(self):
        frame = turned([1, 0, 0])
        assert_allclose(frame.to_world([1, 0, 0]), [1, 1, 0], atol=1e-15)
        assert_allclose(frame.to_local([1, 1, 0]), [1, 0, 0], atol=1e-15)

    def test_grandchild_composes_all_ancestors(self):
        root = turned([1, 0, 0])
        child = CoordinateSystem([1, 0, 0], parent=root)
        grandchild = CoordinateSystem([0, 1, 0], parent=child)
        assert_allclose(grandchild.to_world([1, 0, 0]), [0, 2, 0], atol=1e-15)
        assert_allclose(grandchild.to_local([0, 2, 0]), [1, 0, 0], atol=1e-15)

    def test_deep_hierarchy_has_no_recursion_limit(self):
        root = CoordinateSystem([1, 2, 3])
        leaf = root
        for _ in range(1100):
            leaf = CoordinateSystem(parent=leaf)
        root.translate([1, 0, 0])
        assert_allclose(leaf.to_world([1, 0, 0]), [3, 2, 3])

    def test_axes_stay_orthogonal_at_any_depth(self):
        rng = np.random.default_rng(3)
        frame = CoordinateSystem()
        for _ in range(20):
            frame = CoordinateSystem(parent=frame)
            frame.rotate(rng.normal(size=3), rng.normal())
            frame.scale(rng.choice([-1, 1]) * rng.uniform(0.5, 2))
            frame.translate(rng.normal(size=3))
        axes = np.column_stack([frame.u, frame.v, frame.w])
        s = frame.scale_factor
        assert_allclose(axes.T @ axes, s * s * np.eye(3), atol=1e-12 * s * s)
        assert_allclose(axes, s * frame.rotation.matrix, atol=1e-12)

    def test_world_pose_follows_every_ancestor_change(self):
        root = CoordinateSystem()
        middle = CoordinateSystem([1, 0, 0], parent=root)
        leaf = CoordinateSystem([1, 0, 0], parent=middle)
        other = CoordinateSystem([0, 0, 5])
        changes = [
            (lambda: root.translate([0, 1, 0]), [2, 1, 0]),
            (lambda: middle.rotate([0, 0, 1], 90 * q.deg), [1, 2, 0]),
            (lambda: root.scale(2), [2, 3, 0]),
            (lambda: middle.set_parent(other, keep_world=False), [1, 1, 5]),
            (lambda: other.set_position([0, 0, 0]), [1, 1, 0]),
            (lambda: middle.set_parent(None), [1, 1, 0]),
        ]
        for change, expected in changes:
            change()
            assert_allclose(metres(leaf.origin), expected, atol=1e-14)


class MotionTests(unittest.TestCase):
    def test_translate_relative_to_parent_world_and_local(self):
        for relative_to, expected in [
            ('parent', [0, 3, 0]),
            ('world', [1, 1, 0]),
            ('local', [-2, 1, 0]),
        ]:
            with self.subTest(relative_to=relative_to):
                root = turned(scale=2)
                child = turned([0.5, 0, 0], parent=root)
                child.translate([1, 0, 0], relative_to=relative_to)
                assert_allclose(metres(child.origin), expected, atol=1e-14)

    def test_rotate_about_origin_by_default_and_about_a_pivot(self):
        frame = CoordinateSystem([1, 0, 0])
        frame.add_points([[1, 0, 0]], 'p')
        frame.rotate([0, 0, 1], 90 * q.deg)
        assert_allclose(metres(frame.origin), [1, 0, 0])
        assert_allclose(frame.get_points('p', space='world'), [[1, 1, 0]], atol=1e-15)
        frame.rotate([0, 0, 1], 90 * q.deg, pivot=[0, 0, 0])
        assert_allclose(metres(frame.origin), [0, 1, 0], atol=1e-15)
        assert_allclose(frame.u, [-1, 0, 0], atol=1e-15)

    def test_rotate_axis_relative_to_parent_world_and_local(self):
        parent = Rotation.from_axis_angle([0, 0, 1], 0.9).matrix
        local = Rotation.from_axis_angle([1, 0, 0], 0.3).matrix
        motion = Rotation.from_axis_angle([1, 1, 0], 0.5).matrix
        for relative_to, expected in [
            ('parent', parent @ motion @ local),
            ('world', motion @ parent @ local),
            ('local', parent @ local @ motion),
        ]:
            with self.subTest(relative_to=relative_to):
                root = turned(angle=0.9)
                child = turned(angle=0.3, axis=(1, 0, 0), parent=root)
                child.rotate([1, 1, 0], 0.5, relative_to=relative_to)
                assert_allclose(child.rotation.matrix, expected, atol=1e-15)

    def test_parent_motion_carries_children_and_points(self):
        root = CoordinateSystem([2, 0, 0])
        child = CoordinateSystem([2, 0, 0], parent=root)
        child.add_points([[5, 0, 0]], 'p', space='world')
        grandchild = CoordinateSystem([2, 0, 0], parent=child)
        root.rotate([0, 0, 1], 90 * q.deg, pivot=[1, 0, 0])
        assert_allclose(metres(root.origin), [1, 1, 0], atol=1e-14)
        assert_allclose(metres(child.origin), [1, 3, 0], atol=1e-14)
        assert_allclose(child.get_points('p', space='world'), [[1, 4, 0]], atol=1e-14)
        assert_allclose(metres(grandchild.origin), [1, 5, 0], atol=1e-14)

    def test_scale_keeps_rotation_and_scales_child_offsets(self):
        root = turned([1, 0, 0])
        child = CoordinateSystem([1, 0, 0], parent=root)
        root.scale(2)
        assert_allclose(root.u, [0, 2, 0], atol=1e-15)
        assert_allclose(metres(child.origin), [1, 2, 0], atol=1e-15)
        root.scale(3)
        self.assertAlmostEqual(root.scale_factor, 6)
        assert_allclose(metres(child.origin), [1, 6, 0], atol=1e-14)

    def test_flip_reverses_axes(self):
        frame = turned([1, 2, 0])
        frame.flip()
        assert_allclose(frame.u, [0, -1, 0], atol=1e-15)
        assert_allclose(frame.w, [0, 0, -1], atol=1e-15)
        self.assertAlmostEqual(frame.scale_factor, -1)
        assert_allclose(frame.rotation.matrix, turned().rotation.matrix, atol=1e-15)
        assert_allclose(metres(frame.origin), [1, 2, 0])

    def test_set_position_keeps_child_offsets(self):
        root = CoordinateSystem()
        child = CoordinateSystem([1, 0, 0], parent=root)
        root.set_position([10, 0, 0])
        assert_allclose(metres(child.origin), [11, 0, 0])
        child.set_position([0, 0, 5], relative_to='world')
        assert_allclose(metres(child.origin), [0, 0, 5])
        root.set_position([1 * q.m, 100 * q.cm, 1000 * q.mm])
        assert_allclose(metres(root.origin), [1, 1, 1])

    def test_set_spherical_orients_w_radially_and_keeps_scale(self):
        root = CoordinateSystem(scale=2)
        root.add_points([[1, 0, 0]], 'p')
        child = CoordinateSystem([1, 0, 0], parent=root)
        root.set_spherical(200 * q.cm, 90 * q.deg, 90 * q.deg)
        assert_allclose(metres(root.origin), [0, 2, 0], atol=1e-15)
        assert_allclose(root.u, [0, 0, -2], atol=1e-15)
        assert_allclose(root.v, [-2, 0, 0], atol=1e-15)
        assert_allclose(root.w, [0, 2, 0], atol=1e-15)
        assert_allclose(root.get_points('p', space='world'), [[0, 2, -2]], atol=1e-15)
        assert_allclose(metres(child.origin), [0, 2, -2], atol=1e-15)

    def test_inherit_false_keeps_points_and_children_in_place(self):
        operations = [
            lambda f: f.translate([1, 2, 3], inherit=False),
            lambda f: f.rotate([0, 1, 0], 0.5, pivot=[3, 2, 1], inherit=False),
            lambda f: f.scale(-3, inherit=False),
            lambda f: f.flip(inherit=False),
            lambda f: f.set_position([5, 6, 7], inherit=False),
            lambda f: f.set_spherical(3, 0.4, 0.6, inherit=False),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                root = CoordinateSystem([1, 2, 3], scale=2)
                root.add_points(np.array([[2, 3, 4]]) * q.cm, 'p')
                child = CoordinateSystem([2, 0, 0], parent=root)
                child.add_points([[3, 2, 1]], 'p')
                leaf = CoordinateSystem([4, 5, 6], parent=child)

                def snapshot():
                    return [
                        root.get_points('p', space='world'),
                        child.world_transform.matrix,
                        child.get_points('p', space='world'),
                        leaf.world_transform.matrix,
                    ]

                before = snapshot()
                operation(root)
                for actual, expected in zip(snapshot(), before):
                    assert_allclose(actual, expected, atol=1e-12)

    def test_invalid_input_leaves_the_frame_unchanged(self):
        frame = CoordinateSystem([1, 2, 3])
        before = frame.world_transform.matrix
        for operation in [
            lambda: frame.scale(0),
            lambda: frame.set_spherical(-1, 0, 0),
            lambda: frame.translate([np.inf, 0, 0]),
            lambda: frame.translate([1, 0, 0], relative_to='typo'),
            lambda: frame.rotate([0, 0, 0], 1),
            lambda: frame.translate([1 * q.s, 0, 0]),
        ]:
            with self.assertRaises(ValueError):
                operation()
        assert_allclose(frame.world_transform.matrix, before)

    def test_mixed_units(self):
        root = CoordinateSystem([100, 0, 0], unit=q.cm)
        child = CoordinateSystem([1000, 0, 0], parent=root, unit=q.mm)
        child.add_points([[1000, 0, 0]], 'p')
        root.translate([100, 0, 0])
        assert_allclose(metres(child.origin), [3, 0, 0])
        assert_allclose(child.get_points('p', space='world'), [[4000, 0, 0]])
        assert_allclose(child.to_local([4000, 0, 0]), [1000, 0, 0])
        child.translate(np.array([1, 0, 0]) * q.m)
        assert_allclose(metres(child.origin), [4, 0, 0])


class TreeTests(unittest.TestCase):
    def test_child_pose_is_relative_to_parent_and_inherits_its_unit(self):
        root = turned([100, 0, 0], unit=q.cm)
        child = CoordinateSystem([100, 0, 0], parent=root, name='arm')
        assert_allclose(metres(child.origin), [1, 1, 0], atol=1e-15)
        self.assertEqual(child.unit, q.cm)
        self.assertIs(child.parent, root)
        self.assertEqual(root.children, (child,))
        self.assertIs(root.child('arm'), child)
        self.assertEqual(child.name, 'arm')
        self.assertIn("'arm'", repr(child))

    def test_set_parent_keeps_world_pose_and_moves_attachments(self):
        a, b = CoordinateSystem([1, 2, 3]), turned([4, 5, 6])
        child = CoordinateSystem(parent=a, name='c')
        child.add_points([[1, 0, 0]], 'p')
        before = child.world_transform.matrix, child.get_points('p', space='world')

        child.set_parent(b)
        self.assertEqual(a.children, ())
        self.assertIs(b.child('c'), child)
        assert_allclose(child.world_transform.matrix, before[0], atol=1e-14)
        assert_allclose(child.get_points('p', space='world'), before[1], atol=1e-14)

        child.set_parent(None)
        self.assertIsNone(child.parent)
        self.assertEqual(b.children, ())
        with self.assertRaises(KeyError):
            b.child('c')
        assert_allclose(child.world_transform.matrix, before[0], atol=1e-14)

    def test_set_parent_keep_world_false_reuses_the_local_pose(self):
        parent = CoordinateSystem([10, 0, 0])
        child = CoordinateSystem([1, 0, 0])
        child.set_parent(parent, keep_world=False)
        assert_allclose(metres(child.origin), [11, 0, 0])

    def test_invalid_parents(self):
        a = CoordinateSystem()
        b = CoordinateSystem(parent=a, name='same')
        with self.assertRaises(ValueError):
            a.set_parent(b)
        with self.assertRaises(ValueError):
            a.set_parent(a)
        with self.assertRaises(ValueError):
            CoordinateSystem(parent=a, name='same')
        with self.assertRaises(TypeError):
            b.set_parent('not a frame')
        with self.assertRaises(TypeError):
            CoordinateSystem(rotation=np.eye(3))
        self.assertIs(b.parent, a)
        self.assertEqual(a.children, (b,))

    def test_unnamed_children(self):
        root = CoordinateSystem()
        children = [CoordinateSystem(parent=root) for _ in range(3)]
        self.assertEqual(root.children, tuple(children))
        children[1].set_parent(None)
        self.assertEqual(root.children, (children[0], children[2]))


class PointTests(unittest.TestCase):
    def test_points_are_copied_and_converted_between_spaces(self):
        frame = CoordinateSystem([1, 0, 0])
        data = np.array([[2.0, 0.0, 0.0]])
        frame.add_points(data, 'p', space='world')
        data[:] = 99
        assert_allclose(frame.get_points('p'), [[1, 0, 0]])
        assert_allclose(frame.get_points('p', space='world'), [[2, 0, 0]])
        frame.get_points('p')[:] = 0
        assert_allclose(frame.get_points('p'), [[1, 0, 0]])
        self.assertEqual(frame.point_labels, ('p',))

    def test_points_are_stored_in_the_frame_unit(self):
        frame = CoordinateSystem(unit=q.cm)
        frame.add_points([[1 * q.m, 1 * q.cm, 1 * q.mm]], 'mixed')
        frame.add_points(np.ones((2, 3)) * q.m, 'quantity')
        assert_allclose(frame.get_points('mixed'), [[100, 1, 0.1]])
        self.assertIsInstance(frame.get_points('quantity'), np.ndarray)
        assert_allclose(frame.get_points('quantity'), np.full((2, 3), 100))
        with self.assertRaises(ValueError):
            frame.add_points([[1 * q.s, 0, 0]], 'bad')

    def test_dtype_is_preserved(self):
        frame = CoordinateSystem()
        frame.add_points(np.ones((3, 3), dtype=np.float32), 'p')
        frame.rotate([0, 0, 1], 0.3, inherit=False)
        self.assertEqual(frame.get_points('p').dtype, np.float32)
        self.assertEqual(frame.get_points('p', space='world').dtype, np.float32)

    def test_labels_and_bad_input(self):
        frame = CoordinateSystem()
        frame.add_points([[0, 0, 0]], 'a')
        with self.assertRaises(ValueError):
            frame.add_points([[0, 0, 0]], 'a')
        assert_allclose(frame.remove_points('a'), [[0, 0, 0]])
        with self.assertRaises(KeyError):
            frame.remove_points('a')
        with self.assertRaises(ValueError):
            frame.add_points([[0, 0, 0]], 'b', space='typo')
        with self.assertRaises(ValueError):
            frame.get_points('b', space='typo')
        with self.assertRaises(ValueError):
            frame.add_points([1, 2], 'c')


class AccessorTests(unittest.TestCase):
    def test_accessors_are_snapshots(self):
        frame = CoordinateSystem()
        snapshot = frame.world_transform
        frame.u[:] = 99
        frame.origin[:] = 99 * q.m
        frame.world_transform.matrix[:] = 0
        assert_allclose(frame.to_world([1, 0, 0]), [1, 0, 0])
        frame.translate([1, 0, 0])
        assert_allclose(snapshot.matrix, np.eye(4))

    def test_transforms_use_the_frame_unit(self):
        frame = CoordinateSystem([100, 0, 0], unit=q.cm)
        self.assertEqual(frame.world_transform.unit, q.cm)
        assert_allclose(frame.world_transform.matrix[:3, 3], [100, 0, 0])
        assert_allclose(frame.origin.magnitude, [100, 0, 0])


if __name__ == '__main__':
    unittest.main()
