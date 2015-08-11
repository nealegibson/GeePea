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

#simple wrapper function for solving Toeplitz system in a convenient way
def LTZSolve(a,b,x):
  """
  Ax = b
  a - vector containing the Toeplitz matrix A a->(0-N), 0 being the main diagonal
  b - the input vector (typically the residuals)
  x - the solution space - just an empty matrix, a.size
  """
  #check the vector dimensions are the same
  assert a.size == b.size
  assert a.size == x.size
  
  logdetK = LTZSolveC(a,x,b,a.size)
  return logdetK, x

##########################################################################################

#def CovarianceMatrix(par,X,fixed=None,fixed_par=None,KernelFunction=SqExponentialRad):
def CovarianceMatrixToeplitz(theta,X,ToeplitzKernel):
  """
  Toeplitz equivelent of the CovarianceMatrix function to construct the Toeplitz matrix
  using the same format
  
  """
      
  K = ToeplitzKernel(X,X,theta,white_noise=True)
  
  return K

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
  a = theta[0]**2 * np.exp( -D2 / theta[1]**2 )
  
  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a
  
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

##########################################################################################

#run test
if __name__ == '__main__':
  
  #solve Ax = b, where A is a toeplitz matrix
  # a are the diagonal components of matrix A
  a = np.array([30.,1.,0.3,0.2,0.1,0.2,0.3,15,12,8.])
  b = np.array([1.,1.2,2.3,0.2,1.,2.,1.3,1.,5,.2])
  
  #print the answer from standard/slow inversion
  logdet = np.log(np.linalg.det(LA.toeplitz(a)))
  x = np.mat(LA.toeplitz(a)).I * np.mat(b).T
  print logdet,"\n",np.mat(x)
  
  #and print the LTZ solution
  logdet,x = LTZSolve(a,b)
  print logdet,"\n",x
