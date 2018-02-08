from distutils.core import setup

import numpy
from Cython.Build import cythonize


setup(ext_modules=cythonize("ch_cl.pyx"), include_dirs=[numpy.get_include()]
      )
