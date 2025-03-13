import numpy as np
import quantities as q

# Conditionally import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Backend selection
class Backend:
    NUMPY = 'numpy'
    CUPY = 'cupy'
    
    _current = NUMPY
    
    @classmethod
    def get(cls):
        if cls._current == cls.CUPY and not HAS_CUPY:
            print("Warning: CuPy not available, falling back to NumPy")
            return np
        return cp if cls._current == cls.CUPY else np
    
    @classmethod
    def set(cls, backend):
        if backend == cls.CUPY and not HAS_CUPY:
            print("Warning: CuPy not available, using NumPy instead")
            cls._current = cls.NUMPY
        else:
            cls._current = backend
    
    @classmethod
    def use_gpu(cls):
        cls.set(cls.CUPY)
    
    @classmethod
    def use_cpu(cls):
        cls.set(cls.NUMPY)

    @classmethod
    def to_numpy(cls, array):
        xp = cls.get()
        return array.get() if xp is cp else array
    
    @classmethod
    def to_device(cls, array):
        xp = cls.get()
        return xp.asarray(array)

class Quaternion:
    """
    A class representing a quaternion.
    """
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])
        self.normalized = False
        self._inverse = None

    def real(self):
        return self.q[0]

    def imag(self):
        return self.q[1:] 
    
    def conjugate(self):
        return Quaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])
    
    @staticmethod
    def cross_product(vec1, vec2):
        cross = np.cross(vec1.imag(), vec2.imag())
        return Quaternion(0, *cross)

    @staticmethod
    def dot_product(vec1, vec2):
        dot = np.dot(vec1.imag(), vec2.imag())
        return Quaternion(dot, 0, 0, 0)
    
    @staticmethod
    def scalar_product(vec, scalar):
        return Quaternion(*(vec.q * scalar))
    
    def norm(self):
        return np.linalg.norm(self.q)

    def normalize(self):
        norm = self.norm()
        self.q /= norm
        self.normalized = True
        return self

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.conjugate() / self.norm()
        return self._inverse

    def __add__(self, other):
        return Quaternion(*(self.q + other.q))
    
    def __sub__(self, other):
        return Quaternion(*(self.q - other.q))
    
    def __mul__(self, other):
        # GPU-optimized quaternion multiplication using vectorized operations
        
        # Extracting components in a vectorized way
        w1, v1 = self.q[0], self.q[1:]
        w2, v2 = other.q[0], other.q[1:]
        
        # Computing the components of the resulting quaternion
        w = w1 * w2 - np.dot(v1, v2)
        v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        
        # Create resulting quaternion
        result = np.empty(4, dtype=self.q.dtype)
        result[0] = w
        result[1:] = v
        
        return Quaternion(*result)
    
    def __truediv__(self, scalar):
        return Quaternion(*(self.q / scalar))
    
    def __str__(self):
        # Convert to numpy for string representation if needed
        q_values = Backend.to_numpy(self.q)
        return f"({q_values[0]}, {q_values[1]}, {q_values[2]}, {q_values[3]})"

class RotationQuaternion(Quaternion):
    """
    A class representing a rotation quaternion.
    Non vectorized version.
    """
    def __init__(self, angle, axis):
        self.angle = angle.rescale(q.rad)

        half_angle = self.angle / 2
        
        if len(axis) == 4:
            axis = axis[:3]

        # Convert axis to device array if needed
        axis = Backend.to_device(axis)
        
        sin_half_angle = np.sin(half_angle)
        axis_norm = np.linalg.norm(axis)

        super().__init__(np.cos(half_angle), *(axis * sin_half_angle / axis_norm))

        self.normalized = True
    
    @property
    def matrix(self):
        """
        Homogeneous Orthogonal Matrix representation of the quaternion.
        """
        xp = Backend.get()
        if not self.normalized:
            self.normalize()
        w, x, y, z = self.q
        return xp.array([
            [w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, w*w - x*x + y*y - z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w - x*x - y*y + z*z, 0],
            [0, 0, 0, 1]
        ])
    
    def rotate(self, v):
        # Convert v to device array if needed
        v_device = Backend.to_device(v)
        q = Quaternion(0, *v_device)
        lhs = self
        if not self.normalized:
            rhs = self.inverse()
        else:
            rhs = self.conjugate()

        rotated_vector = (lhs * q * rhs).imag()
        
        try:
            return Backend.to_numpy(rotated_vector) * v.units
        except AttributeError:
            return Backend.to_numpy(rotated_vector)
        

class Transformation(object):
    """
    A class representing a transformation matrix. Uses backend to determine whether to use NumPy or CuPy.
    """
    def __init__(self, matrix=None):
        xp = Backend.get()
        if matrix is None:
            # Initialize with identity matrix
            self.matrix = xp.eye(4, dtype=xp.float64)
        else:
            # Use the provided matrix
            self.matrix = Backend.to_device(matrix)

    def clear(self):
        """
        Clear the transformation matrix.
        """
        self.matrix = Backend.get().eye(4, dtype=Backend.get().float64)

    def apply(self, points):
        """
        Apply transformation to an array of points.
        Points can be a single point [x,y,z] or array of points [[x1,y1,z1], [x2,y2,z2], ...].
        """
        # Store original units if present
        has_units = hasattr(points, 'units')
        units = getattr(points, 'units', None)
        
        xp = Backend.get()
        original_shape = points.shape
        is_single_point = len(original_shape) == 1
        
        # Convert to homogeneous coordinates
        if is_single_point:
            homogeneous_points = xp.ones(4, dtype=xp.float64)
            homogeneous_points[:3] = Backend.to_device(points)
        else:
            num_points = original_shape[0]
            homogeneous_points = xp.ones((num_points, 4), dtype=xp.float64)
            homogeneous_points[:, :3] = Backend.to_device(points)
        
        # Apply transformation
        if is_single_point:
            result = xp.dot(self.matrix, homogeneous_points)
            # Convert back from homogeneous coordinates
            if result[3] != 0 and result[3] != 1:
                result = result / result[3]
            transformed_points = result[:3]
        else:
            result = xp.dot(homogeneous_points, self.matrix.T)
            # Convert back from homogeneous coordinates
            non_zero_w = result[:, 3] != 0
            result[non_zero_w] = result[non_zero_w] / result[non_zero_w, 3:4]
            transformed_points = result[:, :3]
            
        # Return result with original units if applicable
        result = Backend.to_numpy(transformed_points)
        if has_units:
            return result * units
        return result
            
    def compose(self, other):
        """
        Compose this transformation with another transformation.
        Returns a new transformation that applies this transformation followed by the other.
        """
        xp = Backend.get()
        result_matrix = xp.dot(other.matrix, self.matrix)
        return Transformation(result_matrix)
        
    def inverse(self):
        """
        Return the inverse transformation.
        """
        xp = Backend.get()
        inv_matrix = xp.linalg.inv(self.matrix)
        return Transformation(inv_matrix)
        
    @staticmethod
    def translation(x=0.0, y=0.0, z=0.0):
        """
        Create a translation transformation.
        """
        xp = Backend.get()
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return Transformation(matrix)
        
    @staticmethod
    def from_vector(v):
        """
        Create a translation transformation from a vector.
        """
        v = Backend.to_device(v)
        return Transformation.translation(v[0], v[1], v[2])
        
    @staticmethod
    def rotation_x(angle):
        """
        Create a rotation around the x-axis.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = xp.cos(angle_rad), xp.sin(angle_rad)
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[1, 1] = c
        matrix[1, 2] = -s
        matrix[2, 1] = s
        matrix[2, 2] = c
        return Transformation(matrix)
        
    @staticmethod
    def rotation_y(angle):
        """
        Create a rotation around the y-axis.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = xp.cos(angle_rad), xp.sin(angle_rad)
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 0] = c
        matrix[0, 2] = s
        matrix[2, 0] = -s
        matrix[2, 2] = c
        return Transformation(matrix)
        
    @staticmethod
    def rotation_z(angle):
        """
        Create a rotation around the z-axis.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        c, s = xp.cos(angle_rad), xp.sin(angle_rad)
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 0] = c
        matrix[0, 1] = -s
        matrix[1, 0] = s
        matrix[1, 1] = c
        return Transformation(matrix)
        
    @staticmethod
    def rotation(angle, axis):
        """
        Create a rotation around an arbitrary axis.
        Uses RotationQuaternion internally.
        """
        return Transformation(RotationQuaternion(angle, axis).matrix)
        
    @staticmethod
    def scaling(sx=1.0, sy=1.0, sz=1.0):
        """
        Create a scaling transformation.
        """
        xp = Backend.get()
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 0] = sx
        matrix[1, 1] = sy
        matrix[2, 2] = sz
        return Transformation(matrix)
        
    @staticmethod
    def uniform_scaling(scale):
        """
        Create a uniform scaling transformation.
        """
        return Transformation.scaling(scale, scale, scale)
        
    @staticmethod
    def shear_xy(angle):
        """
        Create a shear transformation in the xy plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 1] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def shear_xz(angle):
        """
        Create a shear transformation in the xz plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, 2] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def shear_yx(angle):
        """
        Create a shear transformation in the yx plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[1, 0] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def shear_yz(angle):
        """
        Create a shear transformation in the yz plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[1, 2] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def shear_zx(angle):
        """
        Create a shear transformation in the zx plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[2, 0] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def shear_zy(angle):
        """
        Create a shear transformation in the zy plane.
        """
        xp = Backend.get()
        angle_rad = angle.rescale(q.rad).magnitude if hasattr(angle, 'rescale') else angle
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[2, 1] = xp.tan(angle_rad)
        return Transformation(matrix)
        
    @staticmethod
    def look_at(eye, target, up=[0, 1, 0]):
        """
        Create a viewing transformation that looks from eye to target.
        """
        xp = Backend.get()
        eye = Backend.to_device(eye)
        target = Backend.to_device(target)
        up = Backend.to_device(up)
        
        # Calculate the forward vector (z)
        forward = eye - target
        forward = forward / xp.linalg.norm(forward)
        
        # Calculate the right vector (x)
        right = xp.cross(up, forward)
        right = right / xp.linalg.norm(right)
        
        # Recalculate the up vector (y)
        up = xp.cross(forward, right)
        
        # Create the rotation part
        matrix = xp.eye(4, dtype=xp.float64)
        matrix[0, :3] = right
        matrix[1, :3] = up
        matrix[2, :3] = forward
        
        # Create the translation part
        translation = xp.zeros(3)
        translation[0] = -xp.dot(right, eye)
        translation[1] = -xp.dot(up, eye)
        translation[2] = -xp.dot(forward, eye)
        
        matrix[0:3, 3] = translation
        
        return Transformation(matrix)
        
    def __str__(self):
        matrix_np = Backend.to_numpy(self.matrix)
        return f"Transformation(\n{matrix_np}\n)"


