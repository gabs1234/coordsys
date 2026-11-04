import numpy as np
import quantities as q

from .utils import axis_angle_matrix


class Rotation:
    """An immutable 3D rotation, stored as an orthonormal 3x3 matrix.

    ``a @ b`` is the rotation that applies ``b`` first, then ``a``.
    """

    def __init__(self, matrix=None):
        matrix = np.eye(3) if matrix is None else np.array(matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9):
            raise ValueError('rotation matrix must be an orthonormal 3x3 matrix')
        if np.linalg.det(matrix) < 0:
            raise ValueError('rotation matrix must have determinant +1')
        self._matrix = matrix

    @classmethod
    def _trusted(cls, matrix):
        rotation = cls.__new__(cls)
        rotation._matrix = matrix
        return rotation

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Right-handed rotation about ``axis``; bare angles are radians."""
        return cls._trusted(axis_angle_matrix(axis, angle))

    @classmethod
    def from_quaternion(cls, quaternion):
        """From a quaternion (w, x, y, z); it does not need to be normalized."""
        quaternion = np.asarray(quaternion, dtype=float)
        norm = np.linalg.norm(quaternion)
        if quaternion.shape != (4,) or not np.isfinite(norm) or norm == 0:
            raise ValueError('quaternion must be four finite numbers, not all zero')
        w, x, y, z = quaternion / norm
        return cls._trusted(
            np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
        )

    @property
    def matrix(self):
        return self._matrix.copy()

    @property
    def quaternion(self):
        """The unit quaternion (w, x, y, z), with w >= 0."""
        # Bar-Itzhack: the quaternion is the dominant eigenvector of this matrix.
        (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = self._matrix
        k = np.array(
            [
                [xx + yy + zz, zy - yz, xz - zx, yx - xy],
                [zy - yz, xx - yy - zz, xy + yx, xz + zx],
                [xz - zx, xy + yx, yy - xx - zz, yz + zy],
                [yx - xy, xz + zx, yz + zy, zz - xx - yy],
            ]
        )
        quaternion = np.linalg.eigh(k)[1][:, -1]
        return quaternion if quaternion[0] >= 0 else -quaternion

    def inverse(self):
        return Rotation._trusted(self._matrix.T)

    def apply(self, vectors):
        """Rotate one vector ``(3,)`` or a batch ``(..., 3)``; quantities keep their units."""
        if isinstance(vectors, q.Quantity):
            return self.apply(vectors.magnitude) * vectors.units
        return np.asarray(vectors, dtype=float) @ self._matrix.T

    def __matmul__(self, other):
        if not isinstance(other, Rotation):
            return NotImplemented
        return Rotation._trusted(self._matrix @ other._matrix)

    def __repr__(self):
        return f'Rotation(quaternion={self.quaternion.round(12).tolist()})'
