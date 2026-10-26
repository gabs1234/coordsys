"""Fastplotlib rendering for the coordinate-frame notebook."""

import weakref

import fastplotlib as fpl
import numpy as np
import pygfx
import quantities as pq
from rendercanvas.anywidget import RenderCanvas


def _unit_arrow_geometry():
    """A unit arrow along +z, with a short head joined to a slender shaft."""
    # Fastplotlib 0.7 misplaces the shaft when its height differs from the cone's.
    # Assemble the two standard primitives explicitly, with tail=0 and tip=1.
    shaft = pygfx.cylinder_geometry(
        radius_bottom=0.012,
        radius_top=0.012,
        height=0.82,
        radial_segments=16,
    )
    head = pygfx.cone_geometry(radius=0.045, height=0.18, radial_segments=16)
    shaft.positions.data[:, 2] += 0.41
    head.positions.data[:, 2] += 0.91
    return pygfx.Geometry(
        positions=np.concatenate([shaft.positions.data, head.positions.data]),
        normals=np.concatenate([shaft.normals.data, head.normals.data]),
        indices=np.concatenate(
            [
                shaft.indices.data,
                head.indices.data + len(shaft.positions.data),
            ]
        ),
    )


def _arrow_transform(origin, axis):
    """Place a unit arrow along ``axis``, keeping its proportions for any axis length."""
    length = np.linalg.norm(axis)
    direction = axis / length
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(helper, direction)
    side /= np.linalg.norm(side)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([side, np.cross(direction, side), direction]) * length
    matrix[:3, 3] = origin
    return matrix


class NotebookCanvas(RenderCanvas):
    """Guard rendercanvas 2.7's recursive close notification, without patching it globally."""

    def close(self):
        if getattr(self, '_closing', False) or getattr(self, '_is_closed', False):
            return
        self._closing = True
        try:
            self._rfb_cancel_lossless_draw()
            super().close()
        finally:
            self._closing = False


class GeometryViewer:
    """Keep the notebook's three frames and child point cloud in one live scene.

    Render positions are metres. Only the display buffers use float32;
    CoordinateSystem continues to calculate its transforms at its own precision.
    """

    def __init__(self, points, *, size=(900, 600), canvas=None):
        self.canvas = NotebookCanvas(size=size, max_fps=60) if canvas is None else canvas
        camera = pygfx.OrthographicCamera(8, 6)
        camera.local.position = (6, 6, 6)
        camera.look_at((0, 0, 0))

        # Scope the frame styling to this figure, then restore the caller's defaults.
        config = fpl.layouts.Subplot.config.init
        previous = config.frame_kwargs
        config.frame_kwargs = {
            'spacing': dict(x0=0, sides=0, title_flanks=0, resize_handle_space=0, bottom=0),
            'title_kwargs': dict(font_size=0),
        }
        try:
            self.figure = fpl.Figure(
                cameras=camera,
                controller_types='orbit',
                canvas=self.canvas,
                size=size,
            )
        finally:
            config.frame_kwargs = previous

        self.plot = self.figure[0, 0]
        self.plot.background_color = ('white',)
        self.arrows, self.labels = [], []
        arrow_geometry = _unit_arrow_geometry()
        for name, color in [
            ('Reference', '#666666'),
            ('Main', '#4e79a7'),
            ('Child', '#b86b16'),
        ]:
            arrows = self.plot.add_vectors(
                np.zeros((3, 3), dtype=np.float32),
                np.eye(3, dtype=np.float32),
                color='white',
                size=1,
            )
            arrows.world_object.geometry = arrow_geometry
            arrows.world_object.material.color = color
            self.arrows.append(arrows)
            labels = [
                self.plot.add_text(
                    f'{name}.{axis}',
                    font_size=14,
                    face_color='#222222',
                    screen_space=True,
                    outline_color='white',
                    outline_thickness=0.15,
                )
                for axis in ('u', 'v', 'w')
            ]
            for label in labels:
                label.world_object.material.depth_test = False
            self.labels.append(labels)

        self.cloud = self.plot.add_scatter(
            np.asarray(points, dtype=np.float32),
            colors='#b86b16',
            sizes=7,
            edge_width=0,
        )
        self.link = self.plot.add_line(
            np.zeros((2, 3), dtype=np.float32),
            colors='#666666',
            thickness=1,
        )
        self.figure.show(autoscale=False, axes_visible=False)
        self._bounds = None
        self._cleanup = weakref.finalize(self, self.canvas.close)

    def update(self, reference, main, child):
        """Update poses and local geometry while preserving the user's camera."""
        bounds = []
        for frame, arrows, labels in zip((reference, main, child), self.arrows, self.labels):
            matrix = frame.world_transform.as_unit(pq.m).matrix
            origin, basis = matrix[:3, 3], matrix[:3, :3].T
            bounds.extend([origin, *(origin + 1.12 * basis)])
            for i, (label, axis) in enumerate(zip(labels, basis)):
                arrows.world_object.set_matrix_at(i, _arrow_transform(origin, axis))
                label.offset = origin + 1.12 * axis

        local = (child.get_points('cloud') * child.unit).rescale(pq.m).magnitude
        local = np.asarray(local, dtype=np.float32).reshape(-1, 3)
        if not np.array_equal(local, self.cloud.data[:]):
            self.cloud.data = local
        matrix = child.world_transform.as_unit(pq.m).matrix
        self.cloud.world_object.local.matrix = matrix
        self.link.data[:] = np.array(
            [main.origin.rescale(pq.m).magnitude, child.origin.rescale(pq.m).magnitude],
            dtype=np.float32,
        )
        if local.size:
            bounds.extend(local @ matrix[:3, :3].T + matrix[:3, 3])
        first_update = self._bounds is None
        self._bounds = np.array([np.min(bounds, axis=0), np.max(bounds, axis=0)])
        if first_update:
            self.reset_view()
        self.canvas.request_draw()

    def reset_view(self):
        """Fit the current geometry and restore the initial viewing angle."""
        if self._bounds is None:
            return
        center = self._bounds.mean(axis=0)
        radius = max(float(np.linalg.norm(self._bounds[1] - center)), 1e-6)
        self.plot.camera.show_object(
            (*center, radius),
            view_dir=(-1, 1, -0.8),
            up=(0, 0, 1),
            scale=1.15,
        )
        self.plot.camera.zoom = 1
        self.canvas.request_draw()

    def close(self):
        self._cleanup()
