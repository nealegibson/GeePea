#!/usr/bin/env python

import numpy as np
import scipy.linalg as LA
import ctypes
from numpy.ctypeslib import ndpointer
import os

#import the c function using ctypes
LTZSolveC = ctypes.CDLL('{}/LevinsonTrenchZoharSolve.so'.format(os.path.dirname(__file__))).LevinsonTrenchZoharSolve
#specify the argument and return types
LTZSolveC.argtypes = [ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),\
    ndpointer(ctypes.c_double),ctypes.c_int]
LTZSolveC.restype = ctypes.c_double

##########################################################################################
def ToeplitzSqExponential(X,Y,theta,white_noise=False):
  """
  Toeplitz squared exponential kernel. Must accept arguments in the same format as the
  'normal'/full kernel, but now only returns a vector a describing the diagonal elements
  of the Toeplitz matrix.
  
  To be used as input to LTZSolve
  
  """
  #first calculate distance matrix
  D2 = np.array(np.square(X-Y[0])).flatten()
  
  #calculate Toeplitz 'matrix' stored as array
  a = theta[0]**2 * np.exp( - 0.5 * D2 / theta[1]**2 )
  
  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a

ToeplitzSqExponential.n_par = 3
ToeplitzSqExponential.kernel_type = 'Toeplitz'

##########################################################################################

def ToeplitzMAT_Kernel32(X,Y,theta,white_noise=False):
  """

  Toeplitz Matern covariance kernel for shape =3/2. Must accept arguments in the same
  format as the 'normal'/full kernel, but now only returns a vector a describing the
  diagonal elements of the Toeplitz matrix.
  
  theta[0] - overall scale param - ie prior covariance
  theta[1] - length scale
  theta[2] - white noise


  """

  #first calculate distance matrix
  D = np.array(X-Y[0]).flatten() / theta[1]

  #calculate Toeplitz 'matrix' stored as array
  a = theta[0]**2 * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)

  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a  

ToeplitzMAT_Kernel32.n_par = 3
ToeplitzMAT_Kernel32.kernel_type = 'Toeplitz'

##########################################################################################
