import numpy as np
import cupy as cp
import pyvista as pv
import matplotlib.pyplot as plt
import string
import quantities as q
from .transformations import Quaternion, RotationQuaternion, Transformation
from .utils import X_AX, Y_AX, Z_AX

class CoordinateSystem:
    def __init__(self,
                u : np.array = None,
                v : np.array = None,
                w : np.array = None,
                origin : np.array = None):
        if u is None:
            self._u = X_AX.copy()
        else:
            self._u = u
        if v is None:
            self._v = Y_AX.copy()
        else:
            self._v = v
        if w is None:
            self._w = Z_AX.copy()
        else:
            self._w = w
        if origin is None:
            self._origin = np.array([0, 0, 0], dtype=np.float32) * q.m
        else:
            self._origin = origin
        
        self._children = {}
        self._points = {}
    
    @property
    def u(self):
        return self._u
    
    @u.setter
    def u(self, value):
        self._u = value[:3]
    
    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, value):
        self._v = value[:3]
    
    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, value):
        self._w = value[:3]
    
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, value):
        self._origin = value[:3]

    @property
    def children(self):
        return self._children
    
    @property
    def points(self):
        return self._points
    
    def add_points(self, points, label=None):
        if label is None:
            label = string.ascii_uppercase[len(self._points)]
        self._points[label] = points

    def remove_points(self, label):
        if label in self._points:
            del self._points[label]
            return True
        return False
    
    def get_points(self, label=None):
        if label is None:
            return self._points
        return self._points[label]
    
    def has_child(self, label):
        return label in self._children
    
    def __str__(self):
        return f"({self._origin}, {self._u}, {self._v}, {self._w})"

    def to_opposite(self):
        pass

    def get_local_coordinates(self, point):
        """Get the local coordinates of a point."""
        return point[0] * self._u + point[1] * self._v + point[2] * self._w
    
    def get_global_coordinates(self, point):
        """Get the global coordinates of a point."""
        return self._origin + self.get_local_coordinates(point)

    def get_symmetric(self, r : q.quantity.Quantity, axis=None):
        if axis is None:
            axis = Z_AX
        else:
            axis = np.array(axis[:3])

        local_to_global_origin = self.get_global_coordinates(axis * r)
        self._symmetric = CoordinateSystem(self._u, self._v, self._w, local_to_global_origin)
        return self._symmetric

    def get_antisymmetric(self, r : q.quantity.Quantity, axis=None):
        if axis is None:
            axis = Z_AX
        else:
            axis = np.array(axis[:3])

        opposite = -self._u, -self._v, -self._w
        local_to_global_origin = self.get_global_coordinates(axis * r)
        self._antisymmetric = CoordinateSystem(*opposite, local_to_global_origin)
        return self._antisymmetric
    
    def add_child(self, child=None, label=None):
        if child is None:
            child = CoordinateSystem()
        if label is None:
            label = string.ascii_uppercase[len(self._children)]
        
        # Store a reference to the parent
        child._parent = self
        
        self._children[label] = child

        return child

    def get_parent(self):
        """
        Get the parent coordinate system if this is a child.
        Returns None if this is a root coordinate system.
        """
        return getattr(self, '_parent', None)
    
    def add_symmetric_child(self, label=None, r=None):
        if r is not None:
            d = r
        else:
            d = 1 * q.m
        return self.add_child(self.get_symmetric(d), label=label)
    
    def add_antisymmetric_child(self, label=None, r=None):
        if r is not None:
            d = r
        else:
            d = 1 * q.m
        return self.add_child(self.get_antisymmetric(d), label=label)
    
    def normalize(self):
        self._u = self._u / np.linalg.norm(self._u)
        self._v = self._v / np.linalg.norm(self._v)
        self._w = self._w / np.linalg.norm(self._w)
    
    def translate_along_axis(self, axis, distance, inherit=True, label=None):
        if label is not None:
            self._children[label].translate_along_axis(axis, distance, inherit)
            return
        self._origin += axis * distance

        if inherit:
            # Create a transformation for this translation
            trans = Transformation.from_vector(axis * distance)
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].translate_along_axis(axis, distance, inherit)

    def translate(self, translation, inherit=True, label=None):
        if label is not None:
            self._children[label].translate(translation, inherit)
            return
        self._origin += translation
        
        if inherit:
            # Create a transformation for this translation
            trans = Transformation.from_vector(translation)
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].translate(translation, inherit)

    def set_spherical_coordinates(self, r, theta, phi, inherit=True, label=None):
        if label is not None:
            try:
                self._children[label].set_spherical_coordinates(r, theta, phi, inherit)
            except KeyError:
                pass
            return
        
        cos_theta = np.cos(theta.rescale(q.rad))
        sin_theta = np.sin(theta.rescale(q.rad))
        cos_phi = np.cos(phi.rescale(q.rad))
        sin_phi = np.sin(phi.rescale(q.rad))

        W = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dtype=np.float32)
        U = np.array([cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta], dtype=np.float32)
        V = np.array([-sin_phi, cos_phi, 0], dtype=np.float32)

        # Store original origin
        old_origin = self._origin.copy()
        
        self._u = U
        self._v = V
        self._w = W
        self._origin = r * W

        if inherit:
            # Create a composite transformation
            # 1. First translate back to origin
            trans1 = Transformation.translation(-old_origin[0], -old_origin[1], -old_origin[2])
            # 2. Then apply new position/orientation
            trans2 = Transformation(np.array([
                [U[0], V[0], W[0], self._origin[0]],
                [U[1], V[1], W[1], self._origin[1]],
                [U[2], V[2], W[2], self._origin[2]],
                [0, 0, 0, 1]
            ]))
            # Compose transformations
            trans = trans2.compose(trans1)
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].set_spherical_coordinates(r, theta, phi, inherit)

    def set_cartesian_coordinates(self, x, y, z, inherit=True, label=None):
        if label is not None:
            self._children[label].set_cartesian_coordinates(x, y, z, inherit)
            return
        
        # Store original origin
        old_origin = self._origin.copy()
        
        self._origin = np.array([x, y, z], dtype=np.float32) * x.units

        if inherit:
            # Create a translation transformation
            trans = Transformation.translation(
                self._origin[0] - old_origin[0],
                self._origin[1] - old_origin[1],
                self._origin[2] - old_origin[2]
            )
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].set_cartesian_coordinates(x, y, z, inherit)

    @staticmethod
    def _rotate_vector_euler(vector, axis, angle):
        return RotationQuaternion(angle, axis).rotate(vector)
    
    def rotate_euler(self, pivot, axis, angle, inherit=True, label=None):
        '''
        Rotate the coordinate system about an axis by an angle.
        The pivot is the point about which the rotation occurs.
        If label is not None, the child labeled `label` will be rotated.
        If inherit is True, the children of this coordinate system will also be rotated.
        '''
        if label is not None:
            self._children[label].rotate_euler(pivot, axis, angle, inherit)
            return
            
        # Ensure pivot point has consistent units with the origin
        if hasattr(self._origin, 'units') and hasattr(pivot, 'units'):
            pivot = pivot.rescale(self._origin.units)
        
        # Rotate coordinate system components
        r = self._origin - pivot
        self._origin = self._rotate_vector_euler(r, axis, angle) + pivot
        self._u = self._rotate_vector_euler(self._u, axis, angle)
        self._v = self._rotate_vector_euler(self._v, axis, angle)
        self._w = self._rotate_vector_euler(self._w, axis, angle)

        if inherit:
            # Get root origin 
            parent = self
            while parent.get_parent() is not None:
                parent = self.get_parent()
            root_origin = parent._origin

            # Get global pivot point
            global_pivot = root_origin - pivot
            trans1 = Transformation.translation(-global_pivot[0], -global_pivot[1], -global_pivot[2])
            trans2 = Transformation.rotation(angle, axis)
            trans3 = Transformation.translation(global_pivot[0], global_pivot[1], global_pivot[2])
            trans = trans3.compose(trans2.compose(trans1))
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].rotate_euler(pivot, axis, angle, inherit)
    
    def rotate_euler_local(self, axis, angle, inherit=True, label=None):
        self.rotate_euler(self._origin, axis, angle, inherit, label)
    
    def scale(self, scale : float, inherit=True):
        self._u = X_AX * scale
        self._v = Y_AX * scale
        self._w = Z_AX * scale

        if inherit:
            # Create a scaling transformation relative to origin
            trans1 = Transformation.translation(-self._origin[0], -self._origin[1], -self._origin[2])
            trans2 = Transformation.uniform_scaling(scale)
            trans3 = Transformation.translation(self._origin[0], self._origin[1], self._origin[2])
            trans = trans3.compose(trans2.compose(trans1))
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].scale(scale, inherit)
                self._children[child]._origin = self.get_global_coordinates(self._children[child]._origin)

    def scale_pointwise(self, s1 : float, s2 : float, s3 : float, inherit=True):
        self._u = X_AX * s1
        self._v = Y_AX * s2
        self._w = Z_AX * s3

        if inherit:
            # Create a non-uniform scaling transformation relative to origin
            trans1 = Transformation.translation(-self._origin[0], -self._origin[1], -self._origin[2])
            trans2 = Transformation.scaling(s1, s2, s3)
            trans3 = Transformation.translation(self._origin[0], self._origin[1], self._origin[2])
            trans = trans3.compose(trans2.compose(trans1))
            
            # Transform points
            for key in self._points:
                self._points[key] = trans.apply(self._points[key])
                
            # Transform children
            for child in self._children:
                self._children[child].scale_pointwise(s1, s2, s3, inherit)
                self._children[child]._origin = self.get_global_coordinates(self._children[child]._origin)
    
    def visualize(self, plotter=None, cmap='viridis', inherit=True, show_points=True):
        if plotter is None:
            plotter = pv.Plotter()

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, 3))
        self_origin = self._origin
        self_u = self._u
        self_v = self._v
        self_w = self._w
        plotter.add_arrows(self_origin, self_u, color=colors[0])
        plotter.add_arrows(self_origin, self_v, color=colors[1])
        plotter.add_arrows(self_origin, self_w, color=colors[2])
        plotter.add_points(self_origin, color='black')

        # Plot points if requested
        if show_points and self._points:
            # Use a different colormap for points to distinguish from axes
            point_colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(self._points)))
            
            for i, (label, points) in enumerate(self._points.items()):
                color = point_colors[i % len(point_colors)]
                plotter.add_points(points, color=color, point_size=10, render_points_as_spheres=True)

        if inherit:
            for child in self._children:
                self._children[child].visualize(plotter, cmap=cmap, show_points=show_points)

        return plotter
        
    def get_world_to_local_transformation(self):
        """
        Get a transformation that converts world coordinates to local coordinates.
        """
        # Create rotation matrix from normalized basis vectors
        u_norm = self._u / np.linalg.norm(self._u)
        v_norm = self._v / np.linalg.norm(self._v)
        w_norm = self._w / np.linalg.norm(self._w)
        
        # For world to local conversion:
        # 1. Translate point to align with local origin
        # 2. Rotate using the transposed rotation matrix
        
        # Create rotation matrix (where columns are basis vectors)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 0] = u_norm
        matrix[:3, 1] = v_norm
        matrix[:3, 2] = w_norm
        
        # Create translation matrix to move to local origin
        trans = Transformation.translation(-self._origin[0], -self._origin[1], -self._origin[2])
        
        # Compose the transformations: first translate, then rotate
        return Transformation(matrix.T).compose(trans)
    
    def get_local_to_world_transformation(self):
        """
        Get a transformation that converts local coordinates to world coordinates.
        """
        # Create transformation matrix from basis vectors and origin
        matrix = np.eye(4, dtype=np.float64)
        u_norm = self._u / np.linalg.norm(self._u)
        v_norm = self._v / np.linalg.norm(self._v)
        w_norm = self._w / np.linalg.norm(self._w)
        
        # For local to world conversion:
        # 1. Rotate using the rotation matrix
        # 2. Translate to world origin
        
        # Set columns of rotation matrix to normalized basis vectors
        matrix[:3, 0] = u_norm
        matrix[:3, 1] = v_norm
        matrix[:3, 2] = w_norm
        matrix[:3, 3] = self._origin
        
        return Transformation(matrix)