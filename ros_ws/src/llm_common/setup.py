from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['llm_common'],
    package_dir={'': 'src'},
    requires=['rospy']
)

setup(**d)