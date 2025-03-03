from setuptools import setup, Extension
import numpy as np

setup(

  name = "GeePea", version = "1.0",
  description='python implementation of Gaussian Processes', 
  author='Neale Gibson',
  author_email='n.gibson@tcd.ie',

  packages=['GeePea'],
  package_dir={'GeePea':'src'},
  
  #and extension package for solving toeplitz matrices...
  ext_modules = [
    Extension("GeePea.LevinsonTrenchZoharSolve",sources=["src/LevinsonTrenchZoharSolve.c"]),
    Extension("GeePea.WaveletLikelihood",sources=["src/WaveletLikelihood/WaveletLikelihood.c","src/WaveletLikelihood/FWT.c"],libraries=['gsl','gslcblas']),
    ],

  include_dirs=[np.get_include(),],      

  )
