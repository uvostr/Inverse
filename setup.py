# from setuptools import setup
# from Cython.Build import cythonize

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules=[
    Extension("speedup",
              ["speedup.pyx"],
              include_dirs=[get_include()],
              extra_compile_args=["/O2", "/fp:fast", "/openmp"],
              #extra_link_args=['-openmp']
              )
]

setup(
  name="speedup",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)

