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

The interactive Marimo notebook replaces `README.ipynb`:

```sh
uv run --group notebook marimo edit examples/geometry.py
```

It shows the reference, parent, and child frames in an interactive PyVista scene.
The view has a white background and labels that stay upright as you rotate it.
Rotation and scale controls rebuild the geometry; the world-position table helps
check the result. To open it as an app, use
`uv run --group notebook marimo run examples/geometry.py`.

`uv` installs the requested extras or dependency group automatically. The checked-in
`uv.lock` records the development environment; use `--locked` to require that exact
dependency resolution. Notebook and plotting packages are optional and are not
installed by a plain `uv sync`.

CPU use needs only NumPy and quantities. To use `device='gpu'`, install a CuPy
build matching your CUDA installation. An unavailable GPU raises an error;
`Backend.gpu_available()` lets applications check first.

## Frames

```python
import numpy as np
import quantities as q
from coordsys import CoordinateSystem

world = CoordinateSystem()
sensor = world.add_child(label='sensor')
sensor.add_points([[1, 0, 0]], label='tip', space='local')
sensor.translate([1, 0, 0] * q.m)
sensor.rotate_euler_local([0, 0, 1], 90 * q.deg)

np.testing.assert_allclose(sensor.get_points('tip'), [[1, 1, 0]], atol=1e-14)
np.testing.assert_allclose(sensor.to_local([1, 1, 0]), [1, 0, 0], atol=1e-14)

# A parent move carries the sensor and its points along.
world.translate([0, 2, 0] * q.m)
np.testing.assert_allclose(sensor.get_points('tip'), [[1, 3, 0]], atol=1e-14)
```

Every frame stores **one local-to-parent 4×4 matrix**, a parent link, children,
and local point arrays. Its world matrix is the product along the parent chain:

```text
world_from_child = world_from_parent @ parent_from_child
world_point      = world_from_child @ local_point
```

There is no separately maintained origin, orientation, or world point cache.
`origin`, `u`, `v`, and `w` are derived from that matrix product. This keeps frame
motion and attached geometry consistent, including rotated or scaled ancestors.

| Operation | Meaning |
| --- | --- |
| `to_global(p)` / `to_local(p)` | Inverse conversions between frame and world coordinates. |
| `origin`, `u`, `v`, `w` | World-space position and basis vectors; assigning them moves attached geometry too. |
| `add_child()` | Create a child coincident with its parent. |
| `add_child(existing)` | Attach or reparent while preserving the existing world pose. |
| `add_child(existing, keep_world=False)` | Interpret the existing local pose relative to the new parent. |
| `remove_child(label)` | Detach while preserving world pose. |
| `add_points(p, space='local')` | Attach geometry expressed in the frame's coordinates. |
| `add_points(p)` / `get_points(label)` | World coordinates by default, preserving the older API convention. |
| `translate(v)` | Move along a world-space vector. |
| `rotate_euler(pivot, axis, angle)` | Axis-angle rotation about a world-space pivot and axis. |
| `rotate_euler_local(axis, angle)` | Same rotation about the frame's own origin; the axis is still world-space. |
| `scale(s)` / `scale_pointwise(x, y, z)` | Multiply existing axis lengths about the frame's origin. |
| `set_cartesian_coordinates(x, y, z)` | Set world position; children retain their relative offsets. |
| `set_spherical_coordinates(r, theta, phi)` | Set world position and radial orientation; theta is polar angle, phi azimuth. |

Despite their historical names, the `rotate_euler*` methods take an axis and
angle, not three Euler angles. Numeric angles are radians; quantities may use
degrees. `normalize()` sets axis lengths to one without orthogonalizing a skew
basis. `to_opposite()` reverses all three axes, changing handedness.

Transform methods default to `inherit=True`: attached points and descendants
follow the frame. With `inherit=False`, they remain fixed in world space while
the frame moves. Missing labels raise `KeyError`; duplicate labels, cycles,
and singular frame bases are rejected. Automatic labels extend beyond 26 items.

## Units, precision, and array behavior

Bare coordinates use the object's `unit`, which defaults to metres. Quantity
inputs are converted before arithmetic and returned in their original units:

```python
from coordsys import Transformation

move = Transformation().translation(1 * q.m)
p = np.array([[100, 0, 0]]) * q.cm
np.testing.assert_allclose(move.apply(p).magnitude, [[200, 0, 0]])

centimetres = Transformation(unit=q.cm).translation(100)
np.testing.assert_allclose(centimetres.apply([100, 0, 0]), [200, 0, 0])
```

Frames use metres internally, including when parent and child expose different
units. `local_transform` and `world_transform` return independent transformation
snapshots expressed in the frame's unit; that unit describes both their input
and output numerical coordinates. Quantity inputs avoid any ambiguity.

`float64` is the default. Set `dtype=np.float32` explicitly for smaller point
arrays. Inversion preserves dtype; composition promotes mixed dtypes and
converts length units. Builders apply operations in call order, and
`a.compose(b)` means “apply a, then b.”

`Transformation.apply()` accepts one point `(3,)` or batches `(..., 3)`, including
empty arrays. `vectors=True` omits translation and preserves vector units such
as velocity. General projective matrices are supported for points; `w=0` raises
an error. Coordinate frames require invertible affine matrices.

CPU is the default backend. GPU input arrays stay on the GPU; NumPy inputs and
quantity inputs return to the CPU. Pass `return_numpy=False` to keep the result
on the selected backend, or `True` to request a NumPy result. Use a GPU frame and
CuPy point arrays to keep attached point clouds on the device. Quantities remain
CPU objects.

## Performance and tests

Ordinary frame motion updates a small matrix without visiting descendants or
rewriting point arrays. World transforms are computed along the ancestor chain
when requested. Reuse a `world_transform` snapshot for repeated batches when the
frame is stationary. `inherit=False` must re-express the frame's own points and
direct child transforms to keep their world positions fixed.

Point transforms use `points @ linear.T + translation`, avoiding an `N×4`
homogeneous temporary. Small matrices stay on the CPU; the selected backend
handles bulk array operations. GPU pools are cleared only by an explicit
`Backend.clear_memory()` call.

```sh
uv run python -m unittest discover -s tests -v
uv run python -m performance.run_benchmarks --cpu-only --points 100 10000 1000000 --runs 10
uv run python -m performance.run_benchmarks --test operations --points 10000 --runs 10
uv run --group notebook marimo check examples/geometry.py
```

Tests cover quaternion identities, known geometric results, inverse round trips,
frame inheritance, reparenting, units, dtype, and optional GPU behavior. GPU tests
skip when a working device is unavailable. CI runs the CPU suite on Python 3.10,
3.12, and 3.13.

Benchmarks check results against a float64 homogeneous-coordinate calculation.
They report setup, point upload, warmed array application, and download separately.
GPU application timings synchronize the active stream and retain point arrays on
the device; each call still uploads the small transform matrix. First-use kernel
costs are excluded from the warmed application timing. Unavailable GPUs are omitted.

## Changes from 0.1

The familiar method names remain, with the corrected coordinate conventions
above. Defaults are now CPU and float64. Scaling is multiplicative and preserves
orientation. New children inherit their parent's pose. Accessors return snapshots
and `children` is read-only; use setters and add/remove methods instead of mutating
returned arrays or dictionaries. Plotting dependencies moved to the optional
`visualization` extra.
