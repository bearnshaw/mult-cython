from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("_test",
              ["_test.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],
              include_dirs=[numpy.get_include(), "."]),
]

setup(
    name='test',
    ext_modules=cythonize(extensions),
)
