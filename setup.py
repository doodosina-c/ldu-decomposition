from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "ldu_decomposition.main",
    sources=["ldu_decomposition/main.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    ext_modules=cythonize([ext]),
    packages=["ldu_decomposition"],
)