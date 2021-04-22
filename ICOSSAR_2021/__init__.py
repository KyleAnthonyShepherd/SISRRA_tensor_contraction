'''
This package contains the code I used to generate the ICOSSAR paper

Author
------
Kyle Shepherd, at Rice University

kas20@rice.edu
'''


from setuptools import find_packages
import os
import pkgutil

folder=os.path.dirname(__file__)
__all__ = find_packages(folder)
modules=[name for _, name, _ in pkgutil.iter_modules([folder])]
__all__.extend(modules)
try:
    from . import *
except:
    pass
