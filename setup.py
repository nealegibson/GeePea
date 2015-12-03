from distutils.core import setup, Extension
#from setuptools import setup, Extension

setup(

  name = "GeePea", version = "1.0",
  description='python implementation of Gaussian Processes', 
  author='Neale Gibson',
  author_email='n.gibson@qub.ac.uk',

  packages=['GeePea'],
  package_dir={'GeePea':'src'},
  
  #and extension package for solving toeplitz matrices...
  ext_modules = [
    Extension("GeePea.LevinsonTrenchZoharSolve",sources=["src/LevinsonTrenchZoharSolve.c"]),
    ]

  )
