from setuptools import setup, find_packages

setup(
    name="coordsys",
    version="0.1.0",
    packages=find_packages(),
    description="A Python package for quaternion-based coordinate frame operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gabriel Lefloch",
    author_email="gabriel.lefloch@kit.edu",
    url="https://github.com/gabs1234/coordsys",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pyvista",
        "matplotlib",
        "quantities",
    ],
)