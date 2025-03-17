import pyvista as pv
from coordsys.coordinate_systems import CoordinateSystem
from coordsys.transformations import Backend, Transformation, RotationQuaternion
import numpy as np
import quantities as q

backend = Backend()
backend.use_gpu()
xp = Backend.get()

def generate_random_points(num_points):
    data = xp.random.rand(num_points, 3).astype(xp.float32)
    return Backend.to_numpy(data)

def main():
    reference = CoordinateSystem()
    reference.add_points(generate_random_points(1000), label='random_points')
    
    plotter = pv.Plotter()
    reference.visualize(plotter=plotter)
    plotter.show()

if __name__ == '__main__':
    main()