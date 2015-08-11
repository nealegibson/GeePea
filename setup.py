from distutils.core import setup, Extension
#from setuptools import setup, Extension

setup(

  name = "GeePee", version = "1.0",
  description='Python version of MCMC, plus other GeePeeence codes under development', 
  author='Neale Gibson',
  author_email='ngibson@eso.org',

  packages=['GeePee'],
  package_dir={'GeePee':'src'},
  
  #and extension package for solving toeplitz matrices...
  ext_modules = [
    Extension("GeePee.LevinsonTrenchZoharSolve",sources=["src/LevinsonTrenchZoharSolve.c"]),
    ]

  )
