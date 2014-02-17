"""
Functions to compute the covariance matrix given the data and Kernel Functions
"""

import numpy as np
from GPKernelFunctions import SqExponentialRad #for default kernel function

####################################################################################################

#def CovarianceMatrix(par,X,fixed=None,fixed_par=None,KernelFunction=SqExponentialRad):
def CovarianceMatrix(theta,X,KernelFunction=SqExponentialRad):
  """
  X - input matrix (n x D)
  theta - hyperparameter array/list
  K - (n x n) covariance matrix
  """
  
  #allow fixed parameters and variable params to be passed
#   if fixed_par==None:
#     theta = par[:]
#   else:
#     fixed = np.array(fixed) #ensure fixed is a np array
#     theta = np.empty(fixed.size) #create empty pars array
#     #assign parameters to normal param vector
#     theta[np.where(fixed==True)] = fixed_par
#     theta[np.where(fixed!=True)] = par
    
  K = KernelFunction(X,X,theta,white_noise=True)
  
  return np.matrix(K)

def CovarianceMatrixBlock(theta,X,Y,KernelFunction=SqExponentialRad):
  """
  X - input matrix (q x D) - of training points
  Y - input matrix (n x D) - of precitive points
  theta - hyperparameter array/list
  K - (q x n) covariance matrix block
  """
  
  K = KernelFunction(X,Y,theta,white_noise=False)
  
  return np.matrix(K)

def CovarianceMatrixCornerDiag(theta,X,KernelFunction=SqExponentialRad,WhiteNoise=True):
  """
  X - input matrix (q x D) - of training points
  theta - hyperparameter array/list
  K - (q x q) covariance matrix corner block - only diagonal terms are returned
    (this function needs optimised as it calculates the whole convariance matrix first...)
  """
  
  K = np.diag(np.diag(KernelFunction(X,X,theta,white_noise=WhiteNoise)))
  
  return np.matrix(K)

def CovarianceMatrixCornerFull(theta,X,KernelFunction=SqExponentialRad,WhiteNoise=True):
  """
  X - input matrix (q x D) - of training points
  theta - hyperparameter array/list
  K - (q x q) covariance matrix corner block
  """
  
  K = KernelFunction(X,X,theta,white_noise=WhiteNoise)
  
  return np.matrix(K)

####################################################################################################
