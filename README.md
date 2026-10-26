# Coordsys

Readable 3D coordinate frames, quaternion rotations, and vectorized point
transformations, with physical units through `quantities`.

## Install and run

To add the Git repository as a dependency in another project:

```sh
uv add git+https://github.com/gabs1234/QuaternionFrame.git
```

With [uv](https://docs.astral.sh/uv/), from the repository directory
(Python 3.10 or later):

```sh
uv sync
uv run python examples/simple_transform.py
uv run python -m unittest discover -s tests -v
```

Plotting is optional:

```sh
uv run --extra visualization python examples/simple_transform.py --show
```

The interactive Marimo notebook replaces `README.ipynb` (Python 3.11 or later):

```sh
uv run --group notebook marimo edit examples/geometry.py
```

It shows the reference, parent, and child frames and the child's points with
fastplotlib. The view has a white background and labels that stay upright as you
rotate it. Rotation and scale controls update the geometry while preserving your
camera angle; **Reset view** fits the current setup. The world-position table helps
check the result. To open it as an app, use
`uv run --group notebook marimo run examples/geometry.py`.

The notebook streams rendered images from Python to the browser. It needs a live
Python session and does not produce an interactive standalone HTML export.

`uv` installs the requested extras or dependency group automatically. The checked-in
`uv.lock` records the development environment; use `--locked` to require that exact
dependency resolution. Notebook and plotting packages are optional and are not
installed by a plain `uv sync`.

CPU use needs only NumPy and quantities. Point arrays are processed where they
live: pass CuPy arrays (with a CuPy build matching your CUDA installation) and
results stay on the GPU. `gpu_available()` checks for a working device.

## Frames

```python
import numpy as np
import quantities as q
from coordsys import CoordinateSystem

robot = CoordinateSystem(name='robot')
sensor = CoordinateSystem([1, 0, 0] * q.m, parent=robot, name='sensor')
sensor.add_points([[1, 0, 0]], 'tip')
sensor.rotate([0, 0, 1], 90 * q.deg)

np.testing.assert_allclose(sensor.get_points('tip', space='world'), [[1, 1, 0]], atol=1e-15)
np.testing.assert_allclose(sensor.to_local([1, 1, 0]), [1, 0, 0], atol=1e-15)

# Moving the parent carries the sensor and its points along.
robot.translate([0, 2, 0] * q.m)
np.testing.assert_allclose(sensor.get_points('tip', space='world'), [[1, 3, 0]], atol=1e-15)
```

A `CoordinateSystem` (a *frame*) is an origin, a rotation and one uniform scale
factor. A negative scale reverses all three axes. Its axes `u`, `v` and `w` are
therefore always orthogonal and of equal length, at any depth of the tree.

Each frame stores only its pose relative to its parent, so children and attached
points follow their parent without being updated. Points are stored in the
frame's own coordinates.

| Method | Meaning |
| --- | --- |
| `CoordinateSystem(origin, rotation, scale, parent=None, name=None, unit=None)` | A frame, posed relative to `parent` (or the world). `rotation` is a `Rotation`; `unit` defaults to the parent's, else metres. |
| `set_parent(parent, keep_world=True)` | Attach to another frame, or detach with `None`. By default the world pose is kept. |
| `parent`, `children`, `child(name)`, `name` | Navigate the tree. Names are unique among siblings. |
| `origin`, `rotation`, `scale_factor`, `u`, `v`, `w` | World pose (snapshots). |
| `to_world(p)` / `to_local(p)` | Convert points between this frame and world coordinates. |
| `translate(offset)` | Move by `offset`. |
| `rotate(axis, angle, pivot=None)` | Rotate about `axis` through `pivot` (default: the frame's origin). |
| `set_position(position)` | Place the origin. |
| `set_spherical(r, theta, phi)` | Place the origin in spherical coordinates, with `w` pointing outward. |
| `scale(factor)` / `flip()` | Scale the axes about the origin / reverse them. |
| `add_points(p, label, space='local')` | Attach a copy of points. |
| `get_points(label, space='local')` / `remove_points(label)` / `point_labels` | Read or detach points. |
| `local_transform` / `world_transform` | `Transformation` snapshots of the pose. |

Motion methods take `relative_to='parent'` (default), `'world'` or `'local'`.
Their arguments are coordinates in that frame, so `relative_to='local'` moves
along the frame's own axes, in units of its own (scaled) axis length.

Motion methods also take `inherit=True`. With `inherit=False`, the frame's points
and children stay where they are in the world while the frame moves.

Angles are radians unless given as quantities. Duplicate labels, cycles in the
tree and invalid input raise errors and leave the frame unchanged.

## Transformations

`Transformation` is an immutable affine map. Build one with a constructor and
combine them with `@`, which applies the right-hand side first:

```python
from coordsys import Transformation

move = Transformation.translation([1, 0, 0] * q.m)
turn = Transformation.rotation([0, 0, 1], 90 * q.deg)
both = turn @ move

np.testing.assert_allclose(both.apply([0, 0, 0]), [0, 1, 0], atol=1e-15)
np.testing.assert_allclose(both.inverse().apply([0, 1, 0]), [0, 0, 0], atol=1e-15)
```

Constructors: `translation`, `rotation`, `scaling` (one or three factors),
`shear(plane, angle)`, and `look_at(eye, target, up)`. Use them on point clouds
for maps a frame cannot represent, such as non-uniform scaling or shear.

`apply()` accepts one point `(3,)` or a batch `(..., 3)`. `vectors=True` skips
the translation, for directions and velocities. `Rotation` is the corresponding
rotation-only type (`from_axis_angle`, `from_quaternion`, `quaternion`, `@`).

## Units and arrays

Bare numbers use the object's `unit` (metres by default). Quantities are
converted first, and quantity results come back in their original units:

```python
p = np.array([[100, 0, 0]]) * q.cm
np.testing.assert_allclose(move.apply(p).magnitude, [[200, 0, 0]])

centimetres = Transformation.translation([100, 0, 0], unit=q.cm)
np.testing.assert_allclose(centimetres.apply([100, 0, 0]), [200, 0, 0])
```

Frames compute in metres internally, even when parent and child use different
units. Attached points are stored as plain arrays in the frame's unit, keeping
their dtype (e.g. `float32`) and device (NumPy or CuPy).

## Performance and tests

Moving a frame replaces one 4×4 matrix, without visiting descendants or
rewriting point arrays. World poses are cached and recomputed only for frames
whose ancestors changed. Unit conversion factors are cached too, since
`quantities` conversions are slow. `inherit=False` re-expresses only the
frame's own points and direct children.

Typical costs on a laptop CPU: creating a frame or moving it ~10 µs, reading a
cached world pose ~1 µs, recomputing it after an ancestor moved ~1.5 µs per
level. Point transforms use `points @ linear.T + translation`.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m performance.run_benchmarks --test frames
uv run python -m performance.run_benchmarks --cpu-only --points 100 10000 1000000 --runs 10
uv run python -m performance.run_benchmarks --test operations --points 10000 --runs 10
uv run --group notebook marimo check --strict examples/geometry.py
uv run --group notebook python -m unittest discover -s tests -p test_geometry_viewer.py -v
```

Tests cover rotations, known geometric results, inverse round trips, frame
inheritance, reparenting, units, dtype, and optional GPU behavior. GPU tests
skip when a working device is unavailable. CI runs the CPU suite on Python 3.10,
3.12, and 3.13, plus an offscreen render and geometry checks for the notebook.

Benchmarks check results against a float64 homogeneous-coordinate calculation.
They report setup, point upload, warmed array application, and download separately.
GPU application timings synchronize the active stream and retain point arrays on
the device; each call still uploads the small transform matrix. First-use kernel
costs are excluded from the warmed application timing. Unavailable GPUs are omitted.

## Changes in 0.3

The API was renamed for clarity; the old names are gone.

| 0.2 | 0.3 |
| --- | --- |
| `translate(v)` (world vector) | `translate(v, relative_to='world')`; the default is now `'parent'` |
| `rotate_euler(pivot, axis, angle)` | `rotate(axis, angle, pivot=pivot, relative_to='world')` |
| `rotate_euler_local(axis, angle)` | `rotate(axis, angle, relative_to='world')` |
| `to_global`, `get_global_coordinates` | `to_world` |
| `get_local_coordinates` | `to_local` |
| `set_cartesian_coordinates(x, y, z)` | `set_position([x, y, z])` |
| `set_spherical_coordinates(...)` | `set_spherical(...)` |
| `to_opposite()` | `flip()` |
| `translate_along_axis(axis, d)` | `translate(d * axis)` |
| `get_parent()` | `parent` |
| `add_child(child, label)` | `CoordinateSystem(..., parent=p, name=label)`; the pose is relative to `p` |
| `add_child(existing)` / `remove_child(label)` | `existing.set_parent(p)` / `child.set_parent(None)` |
| `children` (dict of labels) | `children` (tuple), `child(name)` |
| `label=` on motion methods | `frame.child(name).translate(...)` |
| `u`/`v`/`w`/`origin` setters | constructor, `rotate`, `set_position` |
| `scale_pointwise`, `normalize`, skewed axes | `Transformation.scaling` on points |
| `get_symmetric`, `add_symmetric_child`, ... | `CoordinateSystem(offset, parent=p)`, `flip()` |
| `add_points` / `get_points` default `space='world'` | default `space='local'`; `label` is required |
| `Transformation().translation(...)` (mutating) | `Transformation.translation(...)` and `@` |
| `a.compose(b)` | `b @ a` |
| `Quaternion`, `RotationQuaternion` | `Rotation` |
| `Backend`, `device=`, `dtype=` | arrays keep their own device and dtype |

Frames are now restricted to rotation, translation and uniform scale.
