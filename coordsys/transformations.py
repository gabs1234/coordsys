"""Quaternion algebra and vectorized 3D transformations.

Small matrices stay on the CPU. Only point-array operations use the selected
backend; constructing a transform never launches a GPU kernel or clears a pool.
"""

import importlib
import sys

import numpy as np
import quantities as q

from .utils import (
    angle_radians,
    dimensionless,
    float_dtype,
    length_magnitude,
    length_unit,
    unit_vector,
    vector3,
)


def _is_gpu_array(array):
    cupy = sys.modules.get('cupy')
    return cupy is not None and isinstance(array, cupy.ndarray)


class Backend:
    """An explicit NumPy/CuPy choice. CPU is the default; GPU never falls back."""

    NUMPY = 'numpy'
    CUPY = 'cupy'
    _aliases = {'cpu': NUMPY, 'numpy': NUMPY, 'gpu': CUPY, 'cupy': CUPY}

    def __init__(self, device=None):
        self.set('cpu' if device is None else device)

    def set(self, device):
        if not isinstance(device, str) or device.lower() not in self._aliases:
            raise ValueError('device must be cpu, numpy, gpu, or cupy')
        name = self._aliases[device.lower()]
        xp = np
        if name == self.CUPY:
            try:
                xp = importlib.import_module('cupy')
                if xp.cuda.runtime.getDeviceCount() == 0:
                    raise RuntimeError('no CUDA devices')
            except Exception as exc:
                raise RuntimeError('GPU backend requires a working CuPy/CUDA installation') from exc
        self._current, self._xp = name, xp
        return self

    @staticmethod
    def gpu_available():
        try:
            Backend('gpu')
        except RuntimeError:
            return False
        return True

    def get(self):
        return self._xp

    def use_cpu(self):
        return self.set('cpu')

    def use_gpu(self):
        return self.set('gpu')

    @staticmethod
    def to_numpy(array):
        return array.get() if _is_gpu_array(array) else np.asarray(array)

    def to_device(self, array, dtype=None):
        if self._xp is np:
            array = self.to_numpy(array)
        return self._xp.asarray(array, dtype=dtype)

    def synchronize(self):
        if self._current == self.CUPY:
            self._xp.cuda.get_current_stream().synchronize()

    def clear_memory(self):
        """Explicitly release unused GPU allocations; never called implicitly."""
        if self._current == self.CUPY:
            self._xp.get_default_memory_pool().free_all_blocks()
            self._xp.get_default_pinned_memory_pool().free_all_blocks()


class Quaternion:
    """A quaternion (w, x, y, z), with no cached derived state."""

    def __init__(self, w, x, y, z, *, dtype=None):
        self.q = np.array([w, x, y, z], dtype=float_dtype(dtype))
        if self.q.shape != (4,) or not np.all(np.isfinite(self.q)):
            raise ValueError('quaternion components must be finite scalars')

    @property
    def dtype(self):
        return self.q.dtype

    @property
    def normalized(self):
        return bool(np.isclose(self.norm(), 1))

    def real(self):
        return self.q[0]

    def imag(self):
        return self.q[1:].copy()

    def conjugate(self):
        return Quaternion(self.q[0], *(-self.q[1:]), dtype=self.dtype)

    @staticmethod
    def cross_product(vec1, vec2):
        return Quaternion(
            0, *np.cross(vec1.imag(), vec2.imag()), dtype=np.result_type(vec1.dtype, vec2.dtype)
        )

    @staticmethod
    def dot_product(vec1, vec2):
        return Quaternion(
            np.dot(vec1.imag(), vec2.imag()), 0, 0, 0, dtype=np.result_type(vec1.dtype, vec2.dtype)
        )

    @staticmethod
    def scalar_product(vec, scalar):
        return Quaternion(*(vec.q * scalar), dtype=vec.dtype)

    def norm(self):
        return np.linalg.norm(self.q)

    def normalize(self):
        norm = self.norm()
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('cannot normalize a zero or nonfinite quaternion')
        self.q /= norm
        return self

    def inverse(self):
        squared_norm = np.dot(self.q, self.q)
        if not np.isfinite(squared_norm) or squared_norm == 0:
            raise ValueError('cannot invert a zero or nonfinite quaternion')
        return self.conjugate() / squared_norm

    def __add__(self, other):
        return Quaternion(*(self.q + other.q), dtype=np.result_type(self.dtype, other.dtype))

    def __sub__(self, other):
        return Quaternion(*(self.q - other.q), dtype=np.result_type(self.dtype, other.dtype))

    def __mul__(self, other):
        w1, v1 = self.q[0], self.q[1:]
        w2, v2 = other.q[0], other.q[1:]
        return Quaternion(
            w1 * w2 - np.dot(v1, v2),
            *(w1 * v2 + w2 * v1 + np.cross(v1, v2)),
            dtype=np.result_type(self.dtype, other.dtype),
        )

    def __truediv__(self, scalar):
        if np.ndim(scalar) != 0 or not np.isfinite(scalar) or scalar == 0:
            raise ValueError('divisor must be a finite nonzero scalar')
        return Quaternion(*(self.q / scalar), dtype=self.dtype)

    def __str__(self):
        return str(tuple(self.q.tolist()))


class RotationQuaternion(Quaternion):
    """An axis-angle rotation; numeric angles are radians."""

    def __init__(self, angle, axis, *, dtype=None):
        dtype = float_dtype(dtype)
        self.angle = angle_radians(angle)
        axis = unit_vector(axis, dtype=dtype)
        super().__init__(np.cos(self.angle / 2), *(axis * np.sin(self.angle / 2)), dtype=dtype)

    @property
    def matrix(self):
        norm = self.norm()
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('rotation quaternion must be nonzero and finite')
        w, x, y, z = self.q / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )

    def rotate(self, vectors):
        """Rotate one vector or an array of shape (..., 3), retaining units."""
        return Transformation(self.matrix, dtype=self.dtype).apply(vectors, vectors=True)


class Transformation:
    """A 4x4 matrix, a length unit, and a backend for point arrays.

    Mutating builders apply operations in call order. ``a.compose(b)`` returns
    a new transform applying a first, then b. Bare distances use ``unit``;
    quantity inputs are rescaled and returned in their original units.
    """

    def __init__(self, matrix=None, device=None, dtype=None, *, unit=q.m):
        self.dtype = float_dtype(dtype)
        self.unit = length_unit(unit)
        self.backend = Backend(device)
        matrix = np.eye(4) if matrix is None else Backend.to_numpy(matrix)
        self._matrix = np.array(matrix, dtype=self.dtype, copy=True)
        if self._matrix.shape != (4, 4) or not np.all(np.isfinite(self._matrix)):
            raise ValueError('matrix must be a finite 4x4 array')

    @property
    def matrix(self):
        """A copy: change transforms through the builders or constructor."""
        return self._matrix.copy()

    def copy(self):
        return Transformation(self._matrix, self.backend._current, self.dtype, unit=self.unit)

    def reset(self):
        self._matrix = np.eye(4, dtype=self.dtype)
        return self

    def _matrix_in_unit(self, unit, dtype):
        factor = float(self.unit.rescale(unit).magnitude)
        matrix = self._matrix.astype(dtype, copy=True)
        matrix[:3, 3] *= factor
        matrix[3, :3] /= factor
        return matrix

    def as_unit(self, unit):
        """Represent the same physical map using another length unit."""
        unit = length_unit(unit)
        matrix = self._matrix_in_unit(unit, self.dtype)
        return Transformation(matrix, self.backend._current, self.dtype, unit=unit)

    def apply(self, points, *, vectors=False, return_numpy=None):
        """Transform (..., 3) coordinates without an Nx4 temporary.

        ``vectors=True`` omits translation (affine transforms only). GPU inputs
        stay on the GPU by default; NumPy inputs and quantities return to CPU.
        ``return_numpy=False`` keeps a result on the chosen backend explicitly.
        """
        units = points.units if isinstance(points, q.Quantity) else None
        on_gpu = _is_gpu_array(points)
        if units is not None:
            # Direction vectors may carry non-length units, e.g. velocity.
            points = points.magnitude if vectors else points.rescale(self.unit).magnitude
        elif isinstance(points, (list, tuple)):
            # Lists may mix quantities; np.asarray alone would silently strip units.
            points = (
                dimensionless(points, self.dtype)
                if vectors
                else length_magnitude(points, self.unit, self.dtype)
            )
        xp = self.backend.get()
        points = self.backend.to_device(points, self.dtype)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError('points must have shape (..., 3)')
        affine = np.array_equal(self._matrix[3], [0, 0, 0, 1])
        if vectors and not affine:
            raise ValueError('direction vectors require an affine transformation')
        linear = xp.asarray(self._matrix[:3, :3])
        result = points @ linear.T
        if not vectors:
            result += xp.asarray(self._matrix[:3, 3])
            if not affine:
                w = points @ xp.asarray(self._matrix[3, :3]) + self._matrix[3, 3]
                if bool(xp.any(w == 0)):
                    raise ValueError('projective transform maps a point to infinity (w=0)')
                result /= w[..., None]
        if return_numpy is True or (return_numpy is None and not on_gpu) or units is not None:
            result = self.backend.to_numpy(result)
        if units is not None:
            result = result * (units if vectors else self.unit)
            result = result.rescale(units)
        return result

    def compose(self, other):
        """Return a new map applying self first, then other, on self's backend."""
        dtype = np.result_type(self.dtype, other.dtype)
        other_matrix = other._matrix_in_unit(self.unit, dtype)
        return Transformation(
            other_matrix @ self._matrix, self.backend._current, dtype, unit=self.unit
        )

    def inverse(self):
        return Transformation(
            np.linalg.inv(self._matrix), self.backend._current, self.dtype, unit=self.unit
        )

    def _prepend(self, matrix):
        self._matrix = np.asarray(matrix, dtype=self.dtype) @ self._matrix
        return self

    def translation(self, x=0.0, y=0.0, z=0.0):
        matrix = np.eye(4, dtype=self.dtype)
        vector = length_magnitude([x, y, z], self.unit, self.dtype)
        matrix[:3, 3] = vector3(vector, 'translation', self.dtype)
        return self._prepend(matrix)

    def from_vector(self, vector):
        vector = length_magnitude(vector, self.unit, self.dtype)
        return self.translation(*vector3(vector, 'translation', self.dtype))

    def rotation(self, angle, axis):
        return self._prepend(RotationQuaternion(angle, axis, dtype=self.dtype).matrix)

    def rotation_x(self, angle):
        return self.rotation(angle, [1, 0, 0])

    def rotation_y(self, angle):
        return self.rotation(angle, [0, 1, 0])

    def rotation_z(self, angle):
        return self.rotation(angle, [0, 0, 1])

    def scaling(self, sx=1.0, sy=1.0, sz=1.0):
        factors = vector3([sx, sy, sz], 'scale factors', self.dtype)
        return self._prepend(np.diag([*factors, 1]))

    def uniform_scaling(self, scale):
        return self.scaling(scale, scale, scale)

    def _shear(self, row, column, angle):
        matrix = np.eye(4, dtype=self.dtype)
        radians = angle_radians(angle)
        if abs(np.cos(radians)) < np.finfo(self.dtype).eps:
            raise ValueError('shear angle has an undefined tangent')
        matrix[row, column] = np.tan(radians)
        return self._prepend(matrix)

    def shear_xy(self, angle):
        return self._shear(0, 1, angle)

    def shear_xz(self, angle):
        return self._shear(0, 2, angle)

    def shear_yx(self, angle):
        return self._shear(1, 0, angle)

    def shear_yz(self, angle):
        return self._shear(1, 2, angle)

    def shear_zx(self, angle):
        return self._shear(2, 0, angle)

    def shear_zy(self, angle):
        return self._shear(2, 1, angle)

    def look_at(self, eye, target, up=(0, 1, 0)):
        """World-to-view map with eye at zero and target on the negative z-axis."""
        eye = vector3(length_magnitude(eye, self.unit, self.dtype), 'eye', self.dtype)
        target = vector3(length_magnitude(target, self.unit, self.dtype), 'target', self.dtype)
        forward = unit_vector(eye - target, 'eye minus target', self.dtype)
        right = unit_vector(np.cross(unit_vector(up), forward), 'up cross forward', self.dtype)
        matrix = np.eye(4, dtype=self.dtype)
        matrix[:3, :3] = [right, np.cross(forward, right), forward]
        matrix[:3, 3] = -matrix[:3, :3] @ eye
        return self._prepend(matrix)

    def __str__(self):
        return f'Transformation(unit={self.unit}, matrix=\n{self._matrix})'
