# Coordsys : a 3D coordinate system library

A Python package for quaternion-based coordinate frame operations. Supports Quantities package.

## Installation

```bash
pip install -e .
```

## Usage


```python
from coordsys import CoordinateSystem
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

plotter = pv.Plotter()
```


```python
Main = CoordinateSystem()
pl1 = Main.visualize(inherit=False)
pl1.save_graphic('Main.svg')

Child = Main.add_child(label='Child') # Add a child coordinate system to the main one
Child.translate([2, 0, 0]*pq.m)
Child.rotate_euler_local([1, 1, 1], 30*pq.deg)
pl2 = Child.visualize(inherit=False)
pl1.save_graphic('Child.svg')

Main.scale([1, 2, 2]*pq.m, inherit=False) # Scale the main coordinate system, but not its children
pl3 = Main.visualize(inherit=True)
pl1.save_graphic('both.svg')

```

![Parent coordinate system](Main.svg)
![Child coordinate system](Child.svg)
![Both](both.svg)






