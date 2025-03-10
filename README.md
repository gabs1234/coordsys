# QuaternionFrame

A Python package for quaternion-based coordinate frame operations.

## Installation

```bash
pip install -e .
```

## Usage

```python
from quaternion_frame import Quaternion, RotationQuaternion, CoordinateSystem
import quantities as q
import numpy as np

# Create a coordinate system
cs = CoordinateSystem()

# Rotate it
cs.rotate_euler_local(np.array([0, 0, 1]), 45 * q.deg)

# Visualize
cs.visualize()
```
