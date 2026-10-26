"""Frames form a tree; each frame stores only a local-to-parent matrix.

Matrices use metres internally. Attached points stay in local coordinates, so
moving a frame does not traverse its descendants or rewrite its point clouds.
"""

import string
from types import MappingProxyType

import numpy as np
import quantities as q

from .transformations import Backend, Transformation
from .utils import (
    X_AX,
    Y_AX,
    Z_AX,
    angle_radians,
    float_dtype,
    length_magnitude,
    length_unit,
    unit_vector,
    vector3,
)


def _new_label(mapping):
    for index in range(len(mapping) + 1):
        suffix = str(index // 26) if index >= 26 else ''
        label = string.ascii_uppercase[index % 26] + suffix
        if label not in mapping:
            return label


class CoordinateSystem:
    """An invertible affine frame with optional children and point clouds.

    ``origin`` and ``u/v/w`` describe the world pose. Bare coordinates use
    ``unit`` (metres by default); quantity inputs keep their own units.
    Transform methods move the frame, attached points, and descendants together.
    ``inherit=False`` holds the points and descendants fixed in world space.
    """

    def __init__(self, u=None, v=None, w=None, origin=None, device=None, *, dtype=None, unit=q.m):
        self.dtype = float_dtype(dtype)
        self.unit = length_unit(unit)
        self._device = Backend(device)._current
        self._parent = None
        self._children = {}
        self._points = {}
        self._local = np.eye(4, dtype=self.dtype)
        self._local[:3, :3] = np.column_stack(
            [
                vector3(X_AX if u is None else u, 'u', self.dtype),
                vector3(Y_AX if v is None else v, 'v', self.dtype),
                vector3(Z_AX if w is None else w, 'w', self.dtype),
            ]
        )
        if origin is not None:
            self._local[:3, 3] = self._metres(origin)
        self._validate_pose(self._local)

    def _metres(self, value):
        magnitude = length_magnitude(value, self.unit, self.dtype)
        factor = float(self.unit.rescale(q.m).magnitude)
        return vector3(magnitude * factor, 'position', self.dtype)

    @staticmethod
    def _validate_pose(matrix):
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError('frame pose must be a finite 4x4 matrix')
        if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError('coordinate frames require affine transforms')
        if np.linalg.slogdet(matrix[:3, :3])[0] == 0:
            raise ValueError('frame axes must be linearly independent')

    def _world_matrix(self):
        matrix = self._local.copy()
        parent = self._parent
        while parent is not None:
            matrix = parent._local @ matrix
            parent = parent._parent
        return matrix

    def _transform(self, matrix):
        return Transformation(matrix, self._device, matrix.dtype, unit=q.m).as_unit(self.unit)

    @property
    def local_transform(self):
        """A snapshot mapping this frame to its parent, in this frame's unit."""
        return self._transform(self._local)

    @property
    def world_transform(self):
        """A snapshot mapping this frame to world coordinates, in its unit."""
        return self._transform(self._world_matrix())

    def _set_world(self, matrix, inherit=True):
        matrix = np.array(matrix, dtype=self.dtype, copy=True)
        self._validate_pose(matrix)
        local = (
            matrix
            if self._parent is None
            else np.linalg.solve(self._parent._world_matrix(), matrix)
        )
        points, children = self._points, {}
        if not inherit:
            # Re-express only direct attachments. Their world geometry stays put.
            compensation = np.linalg.solve(matrix, self._world_matrix())
            transform = self._transform(compensation)
            points = {key: transform.apply(value) for key, value in self._points.items()}
            children = {key: compensation @ child._local for key, child in self._children.items()}
        self._local = local.astype(self.dtype)
        self._points = points
        for key, child_local in children.items():
            self._children[key]._local = child_local
        return self

    @property
    def origin(self):
        return (self._world_matrix()[:3, 3] * q.m).rescale(self.unit)

    @origin.setter
    def origin(self, value):
        matrix = self._world_matrix()
        matrix[:3, 3] = self._metres(value)
        self._set_world(matrix)

    def _set_axis(self, index, value):
        matrix = self._world_matrix()
        matrix[:3, index] = vector3(value, 'axis', self.dtype)
        self._set_world(matrix)

    @property
    def u(self):
        return self._world_matrix()[:3, 0]

    @u.setter
    def u(self, value):
        self._set_axis(0, value)

    @property
    def v(self):
        return self._world_matrix()[:3, 1]

    @v.setter
    def v(self, value):
        self._set_axis(1, value)

    @property
    def w(self):
        return self._world_matrix()[:3, 2]

    @w.setter
    def w(self, value):
        self._set_axis(2, value)

    @property
    def children(self):
        """Read-only mapping; use add_child/remove_child to change the tree."""
        return MappingProxyType(self._children)

    @property
    def points(self):
        return self.get_points()

    def add_points(self, points, label=None, *, space='world'):
        """Attach a copy of points; world input is the compatibility default."""
        if space not in ('world', 'local'):
            raise ValueError('space must be world or local')
        label = _new_label(self._points) if label is None else label
        if label in self._points:
            raise ValueError(f'point label already exists: {label}')
        if isinstance(points, q.Quantity):
            points.rescale(q.m)  # Validate dimensions without changing display units.
            points = points.astype(self.dtype)
        elif isinstance(points, (list, tuple)):
            points = length_magnitude(points, self.unit, self.dtype)
        else:
            # Preserve GPU residency if the caller supplied a GPU array.
            points = (
                points.astype(self.dtype)
                if hasattr(points, 'astype')
                else np.asarray(points, dtype=self.dtype)
            )
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError('points must have shape (..., 3)')
        self._points[label] = self.to_local(points) if space == 'world' else points.copy()
        return label

    def remove_points(self, label):
        if label not in self._points:
            return False
        del self._points[label]
        return True

    def get_points(self, label=None, *, space='world'):
        """Return a point cloud (or dictionary of clouds), as independent arrays."""
        if space not in ('world', 'local'):
            raise ValueError('space must be world or local')
        convert = self.world_transform.apply if space == 'world' else lambda value: value.copy()
        if label is not None:
            return convert(self._points[label])
        return {key: convert(value) for key, value in self._points.items()}

    def to_global(self, points):
        return self.world_transform.apply(points)

    def to_local(self, points):
        return self.world_transform.inverse().apply(points)

    get_global_coordinates = to_global
    get_local_coordinates = to_local

    def get_parent(self):
        return self._parent

    def has_child(self, label):
        return label in self._children

    def add_child(self, child=None, label=None, *, keep_world=True):
        """Create a coincident child, or attach/reparent an existing frame.

        Existing frames keep their world pose by default. With keep_world=False,
        their previous local matrix is used relative to this parent instead.
        """
        label = _new_label(self._children) if label is None else label
        if label in self._children:
            raise ValueError(f'child label already exists: {label}')
        if child is None:
            child = CoordinateSystem(device=self._device, dtype=self.dtype, unit=self.unit)
            keep_world = False
        if not isinstance(child, CoordinateSystem):
            raise TypeError('child must be a CoordinateSystem')
        parent = self
        while parent is not None:
            if parent is child:
                raise ValueError('a frame cannot be its own ancestor')
            parent = parent._parent
        local = (
            np.linalg.solve(self._world_matrix(), child._world_matrix())
            if keep_world
            else child._local.copy()
        )
        # Compute everything that can fail before changing either tree.
        if child._parent is not None:
            old_parent = child._parent
            old_label = next(key for key, value in old_parent._children.items() if value is child)
            del old_parent._children[old_label]
        child._local = local
        child._parent = self
        self._children[label] = child
        return child

    def remove_child(self, label):
        """Detach a child, preserving its world pose and attached geometry."""
        child = self._children[label]
        world = child._world_matrix()
        del self._children[label]
        child._parent, child._local = None, world
        return child

    def translate(self, translation, inherit=True, label=None):
        """Translate along a world-space vector."""
        if label is not None:
            return self._children[label].translate(translation, inherit)
        matrix = self._world_matrix()
        matrix[:3, 3] += self._metres(translation)
        return self._set_world(matrix, inherit)

    def translate_along_axis(self, axis, distance, inherit=True, label=None):
        if label is not None:
            return self._children[label].translate_along_axis(axis, distance, inherit)
        return self.translate(unit_vector(axis) * distance, inherit)

    def set_cartesian_coordinates(self, x, y, z, inherit=True, label=None):
        if label is not None:
            return self._children[label].set_cartesian_coordinates(x, y, z, inherit)
        matrix = self._world_matrix()
        matrix[:3, 3] = self._metres([x, y, z])
        return self._set_world(matrix, inherit)

    def set_spherical_coordinates(self, r, theta, phi, inherit=True, label=None):
        """Set world pose: theta is polar angle, phi azimuth; w points radially."""
        if label is not None:
            return self._children[label].set_spherical_coordinates(r, theta, phi, inherit)
        radius = length_magnitude(r, self.unit)
        if radius.shape != () or not np.isfinite(radius) or radius < 0:
            raise ValueError('radius must be a finite nonnegative scalar')
        theta, phi = angle_radians(theta), angle_radians(phi)
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        matrix = np.eye(4, dtype=self.dtype)
        matrix[:3, :3] = np.array([[cp * ct, -sp, cp * st], [sp * ct, cp, sp * st], [-st, 0, ct]])
        matrix[:3, 3] = self._metres(radius * matrix[:3, 2])
        return self._set_world(matrix, inherit)

    def rotate_euler(self, pivot, axis, angle, inherit=True, label=None):
        """Legacy name for an axis-angle rotation; pivot and axis are in world space."""
        if label is not None:
            return self._children[label].rotate_euler(pivot, axis, angle, inherit)
        pivot = self._metres(pivot)
        transform = Transformation(dtype=self.dtype).rotation(angle, axis).matrix
        transform[:3, 3] = pivot - transform[:3, :3] @ pivot
        return self._set_world(transform @ self._world_matrix(), inherit)

    def rotate_euler_local(self, axis, angle, inherit=True, label=None):
        """Rotate about this frame's origin; axis is still a world-space vector."""
        if label is not None:
            return self._children[label].rotate_euler_local(axis, angle, inherit)
        return self.rotate_euler(self.origin, axis, angle, inherit)

    def scale(self, scale, inherit=True):
        return self.scale_pointwise(scale, scale, scale, inherit)

    def scale_pointwise(self, s1, s2, s3, inherit=True):
        """Multiply the existing local axes about this frame's origin."""
        factors = vector3([s1, s2, s3], 'scale factors', self.dtype)
        matrix = self._world_matrix()
        matrix[:3, :3] *= factors
        return self._set_world(matrix, inherit)

    def normalize(self, inherit=True):
        """Give each axis unit length; this does not orthogonalize a skew frame."""
        lengths = np.linalg.norm(self._world_matrix()[:3, :3], axis=0)
        return self.scale_pointwise(*(1 / lengths), inherit=inherit)

    def to_opposite(self, inherit=True):
        """Reverse all three axes (a reflection, changing handedness)."""
        return self.scale(-1, inherit)

    def get_symmetric(self, r, axis=None):
        """Return an independent frame displaced along a local axis (default z)."""
        axis = unit_vector(Z_AX if axis is None else axis)
        offset = axis * length_magnitude(r, self.unit)
        return CoordinateSystem(
            self.u,
            self.v,
            self.w,
            self.to_global(offset),
            self._device,
            dtype=self.dtype,
            unit=self.unit,
        )

    def get_antisymmetric(self, r, axis=None):
        return self.get_symmetric(r, axis).to_opposite()

    def add_symmetric_child(self, label=None, r=None):
        return self.add_child(self.get_symmetric(1 * self.unit if r is None else r), label)

    def add_antisymmetric_child(self, label=None, r=None):
        return self.add_child(self.get_antisymmetric(1 * self.unit if r is None else r), label)

    def visualize(self, plotter=None, cmap='viridis', inherit=True, show_points=True):
        """Draw world geometry in this frame's unit; plotting imports are optional."""
        try:
            import pyvista as pv
            from matplotlib import colormaps
        except ImportError as exc:
            raise ImportError("visualize requires pip install 'coordsys[visualization]'") from exc
        plotter = pv.Plotter() if plotter is None else plotter
        colors = colormaps[cmap](np.linspace(0, 1, 3))
        pending = [self]
        while pending:
            frame = pending.pop()
            world = frame._world_matrix()
            origin = (world[:3, 3] * q.m).rescale(self.unit).magnitude
            for index, color in enumerate(colors):
                plotter.add_arrows(origin, world[:3, index], color=color)
            plotter.add_points(origin, color='black')
            if show_points:
                for index, points in enumerate(frame.get_points().values()):
                    if not isinstance(points, q.Quantity):
                        points = Backend.to_numpy(points) * frame.unit
                    points = points.rescale(self.unit).magnitude.reshape(-1, 3)
                    if len(points):
                        plotter.add_points(
                            points,
                            color=colormaps['tab10'](index % 10),
                            point_size=10,
                            render_points_as_spheres=True,
                        )
            if inherit:
                pending.extend(reversed(tuple(frame._children.values())))
        return plotter

    def __str__(self):
        return f'CoordinateSystem(origin={self.origin}, u={self.u}, v={self.v}, w={self.w})'
