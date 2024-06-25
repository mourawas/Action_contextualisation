from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[],
    package_dir={'': 'src'},
    requires=['rospy', 'llm_simulator', 'llm_common', 'iiwa_tools']
)

setup(**d)