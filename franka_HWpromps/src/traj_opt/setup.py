#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['traj_opt'],
    package_dir={'': 'scripts'},
    requires=['numpy', 'rospkg']
)

setup(**setup_args)


