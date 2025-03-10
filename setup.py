from setuptools import setup, find_packages

setup(
    name="quaternion_frame",
    version="0.1.0",
    packages=find_packages(),
    description="A package for quaternion-based coordinate frame operations",
    author="Author",
    install_requires=[
        "numpy",
        "pyvista",
        "matplotlib",
        "quantities",
    ],
)
