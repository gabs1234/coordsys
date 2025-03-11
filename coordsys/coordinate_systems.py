import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import string
import quantities as q
from .quaternions import Quaternion, RotationQuaternion
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
        self._children[label] = child

        return child
    
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
            for child in self._children:
                self._children[child].translate_along_axis(axis, distance, inherit)

    def translate(self, translation, inherit=True, label=None):
        if label is not None:
            self._children[label].translate(translation, inherit)
            return
        self._origin += translation

        if inherit:
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

        self._u = U
        self._v = V
        self._w = W
        self._origin = r * W

        if inherit:
            for child in self._children:
                self._children[child].set_spherical_coordinates(r, theta, phi, inherit)

    def set_cartesian_coordinates(self, x, y, z, inherit=True, label=None):
        if label is not None:
            self._children[label].set_cartesian_coordinates(x, y, z, inherit)
            return
        
        self._origin = np.array([x, y, z], dtype=np.float32) * x.units

        if inherit:
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
        r = self._origin - pivot
        self._origin = self._rotate_vector_euler(r, axis, angle) + pivot
        self._u = self._rotate_vector_euler(self._u, axis, angle)
        self._v = self._rotate_vector_euler(self._v, axis, angle)
        self._w = self._rotate_vector_euler(self._w, axis, angle)

        if inherit:
            for child in self._children:
                self._children[child].rotate_euler(pivot, axis, angle, inherit)

    def rotate_euler_local(self, axis, angle, inherit=True, label=None):
        self.rotate_euler(self._origin, axis, angle, inherit, label)
    
    def scale(self, scale : float, inherit=True):
        self._u = X_AX * scale
        self._v = Y_AX * scale
        self._w = Z_AX * scale

        if inherit:
            for child in self._children:
                self._children[child].scale(scale, inherit)
                self._children[child]._origin = self.get_global_coordinates(self._children[child]._origin)

    def scale_pointwise(self, s1 : float, s2 : float, s3 : float, inherit=True):
        self._u = X_AX * s1
        self._v = Y_AX * s2
        self._w = Z_AX * s3

        if inherit:
            for child in self._children:
                self._children[child].scale_pointwise(s1, s2, s3, inherit)
                self._children[child]._origin = self.get_global_coordinates(self._children[child]._origin)
    
    def visualize(self, plotter=None, cmap='viridis', inherit=True):
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

        if inherit:
            for child in self._children:
                self._children[child].visualize(plotter, cmap=cmap)
        
        if plotter is None:
            plotter.show()

        return plotter
