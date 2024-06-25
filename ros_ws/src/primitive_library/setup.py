from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['controller_base', 'primitives', 'neural_jsdf'],
    package_dir={'': 'src'},
    requires=['rospy']
)

setup(**d)