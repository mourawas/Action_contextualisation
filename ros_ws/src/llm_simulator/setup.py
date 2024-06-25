from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['controller_utils', 'tools', 'kinematics', 'description'],
    package_dir={'': 'src'},
    requires=['rospy', 'llm_common']
)

setup(**d)