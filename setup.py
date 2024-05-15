from setuptools import find_packages
from distutils.core import setup

setup(
    name="QGym",
    version="1.0.3",
    author="Biomimetic Robotics Lab",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Isaac Gym environments",
    install_requires=[
        "isaacgym",
        "setuptools==59.5.0",
        "torch>=1.4.0",
        "torchvision>=0.5.0",
    ],
)
