import numpy as np
import quantities as q

from .backend import float_points, to_numpy
from .rotation import Rotation
from .transformations import Transformation
from .utils import (
    Y_AX,
    Z_AX,
    axis_angle_matrix,
    conversion_factor,
    finite_scalar,
    homogeneous,
    length_unit,
    magnitude,
    vector3,
)

REFERENCES = ('parent', 'world', 'local')
SPACES = ('local', 'world')
IDENTITY = np.eye(4)
IDENTITY.flags.writeable = False


def _inverse(pose):
    """Inverse of a 4x4 pose ``[[s R, t], [0, 1]]``, without a general matrix inversion."""
    linear = pose[:3, :3]
    inverse_linear = linear.T / (linear[:, 0] @ linear[:, 0])
    return homogeneous(inverse_linear, -inverse_linear @ pose[:3, 3])


class CoordinateSystem:
    """A coordinate frame: an origin, an orientation and a uniform scale.

    Frames form a tree. Each frame stores its pose relative to its parent (or
    to the world for a root), so children and attached points follow it.
    Its axes ``u``, ``v`` and ``w`` are always orthogonal and of equal length.

    Bare numbers are read in ``unit`` (the parent's unit by default, else
    metres); quantities may use any length unit.

    Motion methods take ``relative_to='parent' | 'world' | 'local'``: their
    arguments are coordinates in that frame. With ``inherit=False``, the
    frame's points and children stay where they are in the world.
    """

    def __init__(
        self, origin=(0, 0, 0), rotation=None, scale=1.0, *, parent=None, name=None, unit=None
    ):
        if unit is None and parent is not None:
            self.unit, self._metres_per_unit = parent.unit, parent._metres_per_unit
        else:
            self.unit = length_unit(q.m if unit is None else unit)
            self._metres_per_unit = conversion_factor(self.unit, q.m)
        self._name = name
        self._parent = None
        self._children = {}
        self._named_children = {}
        self._points = {}
        self._cached_parent_world = self._cached_local = self._cached_world = None

        if rotation is not None and not isinstance(rotation, Rotation):
            raise TypeError('rotation must be a Rotation')
        scale = self._scale_factor(scale)
        linear = np.eye(3) if rotation is None else rotation.matrix
        self._local = homogeneous(scale * linear, self._metres(origin, 'origin'))
        if parent is not None:
            self.set_parent(parent, keep_world=False)

    # Tree

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return tuple(self._children)

    def child(self, name):
        return self._named_children[name]

    def set_parent(self, parent, *, keep_world=True):
        """Attach this frame to ``parent``, or detach it with ``None``.

        By default the frame keeps its world pose. With ``keep_world=False``
        its current pose is reinterpreted relative to the new parent.
        """
        if parent is not None:
            if not isinstance(parent, CoordinateSystem):
                raise TypeError('parent must be a CoordinateSystem or None')
            ancestor = parent
            while ancestor is not None:
                if ancestor is self:
                    raise ValueError('a frame cannot be its own ancestor')
                ancestor = ancestor._parent
            sibling = parent._named_children.get(self._name)
            if sibling is not None and sibling is not self:
                raise ValueError(f'{parent!r} already has a child named {self._name!r}')

        if keep_world:
            world = self._world()
            self._local = world.copy() if parent is None else _inverse(parent._world()) @ world
        if self._parent is not None:
            del self._parent._children[self]
            self._parent._named_children.pop(self._name, None)
        if parent is not None:
            parent._children[self] = None
            if self._name is not None:
                parent._named_children[self._name] = self
        self._parent = parent
        return self

    # Pose

    def _world(self):
        """World pose as a read-only 4x4 matrix in metres.

        Poses are replaced, never modified in place, so a cached world pose is
        valid while its inputs (parent world pose, local pose) are the same objects.
        """
        chain, frame = [], self
        while frame is not None:
            chain.append(frame)
            frame = frame._parent
        world = IDENTITY
        for frame in reversed(chain):
            if frame._cached_parent_world is not world or frame._cached_local is not frame._local:
                frame._cached_parent_world, frame._cached_local = world, frame._local
                frame._cached_world = world @ frame._local
                frame._cached_world.flags.writeable = False
            world = frame._cached_world
        return world

    def _parent_world(self):
        return IDENTITY if self._parent is None else self._parent._world()

    def _in_unit(self, pose):
        if self._metres_per_unit == 1:
            return Transformation._trusted(pose.copy(), self.unit)
        matrix = pose.copy()
        matrix[:3, 3] /= self._metres_per_unit
        return Transformation._trusted(matrix, self.unit)

    @property
    def local_transform(self):
        """Snapshot of the map from this frame's coordinates to its parent's, in ``unit``."""
        return self._in_unit(self._local)

    @property
    def world_transform(self):
        """Snapshot of the map from this frame's coordinates to world coordinates, in ``unit``."""
        return self._in_unit(self._world())

    @property
    def origin(self):
        return self._world()[:3, 3] / self._metres_per_unit * self.unit

    @property
    def scale_factor(self):
        return float(np.cbrt(np.linalg.det(self._world()[:3, :3])))

    @property
    def rotation(self):
        return Rotation._trusted(self._world()[:3, :3] / self.scale_factor)

    @property
    def u(self):
        return self._world()[:3, 0].copy()

    @property
    def v(self):
        return self._world()[:3, 1].copy()

    @property
    def w(self):
        return self._world()[:3, 2].copy()

    # Coordinates

    def to_world(self, points):
        """Convert points from this frame's coordinates to world coordinates."""
        return self.world_transform.apply(points)

    def to_local(self, points):
        """Convert points from world coordinates to this frame's coordinates."""
        return self._in_unit(_inverse(self._world())).apply(points)

    # Motion

    def translate(self, offset, *, relative_to='parent', inherit=True):
        motion = homogeneous(np.eye(3), self._metres(offset, 'offset'))
        return self._move(motion, relative_to, inherit)

    def rotate(self, axis, angle, *, pivot=None, relative_to='parent', inherit=True):
        """Rotate about ``axis`` through ``pivot`` (default: this frame's origin)."""
        rotation = axis_angle_matrix(axis, angle)
        if pivot is None:
            pivot = self._pose_in(relative_to)[:3, 3]
        else:
            pivot = self._metres(pivot, 'pivot')
        return self._move(homogeneous(rotation, pivot - rotation @ pivot), relative_to, inherit)

    def scale(self, factor, *, inherit=True):
        """Multiply the axis lengths by ``factor``, about this frame's origin."""
        motion = homogeneous(self._scale_factor(factor) * np.eye(3))
        return self._move(motion, 'local', inherit)

    def flip(self, *, inherit=True):
        """Reverse all three axes. This changes the frame's handedness."""
        return self.scale(-1, inherit=inherit)

    def set_position(self, position, *, relative_to='parent', inherit=True):
        target = self._pose_in(relative_to).copy()
        target[:3, 3] = self._metres(position, 'position')
        return self._place(target, relative_to, inherit)

    def set_spherical(self, r, theta, phi, *, relative_to='parent', inherit=True):
        """Place the origin at (r, theta, phi), with ``w`` pointing radially outward.

        ``theta`` is the polar angle from z and ``phi`` the azimuth from x.
        ``u`` points towards increasing theta and ``v`` towards increasing phi.
        """
        radius = finite_scalar(r, 'r', self.unit) * self._metres_per_unit
        if radius < 0:
            raise ValueError('r must be nonnegative')
        rotation = axis_angle_matrix(Z_AX, phi) @ axis_angle_matrix(Y_AX, theta)
        scale = np.cbrt(np.linalg.det(self._pose_in(relative_to)[:3, :3]))
        target = homogeneous(scale * rotation, radius * rotation[:, 2])
        return self._place(target, relative_to, inherit)

    def _pose_in(self, relative_to):
        """This frame's pose, in the coordinates of the ``relative_to`` frame."""
        if relative_to == 'parent':
            return self._local
        if relative_to == 'world':
            return self._world()
        if relative_to == 'local':
            return IDENTITY
        raise ValueError(f'relative_to must be one of {REFERENCES}')

    def _place(self, target, relative_to, inherit):
        """Give this frame the pose ``target``, expressed in ``relative_to`` coordinates."""
        if relative_to == 'world':
            target = _inverse(self._parent_world()) @ target
        elif relative_to == 'local':
            target = self._local @ target
        return self._set_local(target, inherit)

    def _move(self, motion, relative_to, inherit):
        """Apply ``motion``, expressed in ``relative_to`` coordinates."""
        return self._place(motion @ self._pose_in(relative_to), relative_to, inherit)

    def _set_local(self, local, inherit):
        if not inherit:
            correction = _inverse(local) @ self._local
            for child in self._children:
                child._local = correction @ child._local
            transform = self._in_unit(correction)
            self._points = {label: transform.apply(value) for label, value in self._points.items()}
        self._local = local
        return self

    def _metres(self, value, name):
        return vector3(value, name, self.unit) * self._metres_per_unit

    @staticmethod
    def _scale_factor(value):
        value = finite_scalar(value, 'scale')
        if value == 0:
            raise ValueError('scale must be nonzero')
        return value

    # Points

    def add_points(self, points, label, *, space='local'):
        """Attach a copy of ``points`` (shape ``(..., 3)``) under ``label``.

        Points are stored in this frame's coordinates and unit, keeping their
        dtype and device, and move with the frame.
        """
        if space not in SPACES:
            raise ValueError(f'space must be one of {SPACES}')
        if label in self._points:
            raise ValueError(f'point label already exists: {label!r}')
        if isinstance(points, (q.Quantity, list, tuple)):
            points = magnitude(points, self.unit)
        points = float_points(points)
        self._points[label] = self.to_local(points) if space == 'world' else points.copy()

    def remove_points(self, label):
        """Detach the points stored under ``label`` and return them, in local coordinates."""
        return self._points.pop(label)

    def get_points(self, label, *, space='local'):
        """A copy of the points stored under ``label``, in this frame's unit."""
        if space not in SPACES:
            raise ValueError(f'space must be one of {SPACES}')
        points = self._points[label]
        return self.to_world(points) if space == 'world' else points.copy()

    @property
    def point_labels(self):
        return tuple(self._points)

    # Display

    def visualize(self, plotter=None, cmap='viridis', children=True, show_points=True):
        """Draw the frame (and its descendants) in world coordinates with PyVista."""
        try:
            import pyvista as pv
            from matplotlib import colormaps
        except ImportError as exc:
            raise ImportError("visualize requires pip install 'pyframetree[visualization]'") from exc

        plotter = pv.Plotter() if plotter is None else plotter
        axis_colors = colormaps[cmap](np.linspace(0, 1, 3))
        frames = [self]
        while frames:
            frame = frames.pop()
            origin = frame.origin.rescale(self.unit).magnitude
            for axis, color in zip((frame.u, frame.v, frame.w), axis_colors):
                plotter.add_arrows(origin, axis, color=color)
            plotter.add_points(origin, color='black')
            if show_points:
                for index, label in enumerate(frame._points):
                    points = to_numpy(frame.get_points(label, space='world'))
                    points = points * conversion_factor(frame.unit, self.unit)
                    if points.size:
                        plotter.add_points(
                            points.reshape(-1, 3),
                            color=colormaps['tab10'](index % 10),
                            point_size=10,
                            render_points_as_spheres=True,
                        )
            if children:
                frames.extend(reversed(frame.children))
        return plotter

    def __repr__(self):
        name = '' if self._name is None else f'{self._name!r}, '
        return f'CoordinateSystem({name}origin={self.origin}, scale={self.scale_factor:g})'
