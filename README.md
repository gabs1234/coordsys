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
Main.visualize(inherit=False).show()

Child = Main.add_child(label='Child') # Add a child coordinate system to the main one
Child.translate([2, 0, 0]*pq.m)
Child.rotate_euler_local([1, 1, 1], 30*pq.deg)
Child.visualize(inherit=False).show()

Main.scale([1, 2, 2]*pq.m, inherit=False) # Scale the main coordinate system, but not its children
Main.visualize(inherit=True).show()

```


    Widget(value='<iframe src="http://localhost:39627/index.html?ui=P_0x70139ff10bf0_11&reconnect=auto" class="pyv…



    Widget(value='<iframe src="http://localhost:39627/index.html?ui=P_0x70139740f710_12&reconnect=auto" class="pyv…



    Widget(value='<iframe src="http://localhost:39627/index.html?ui=P_0x701397408380_13&reconnect=auto" class="pyv…



```python

```
