from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("brute_force_cython_ext.pyx")
)
