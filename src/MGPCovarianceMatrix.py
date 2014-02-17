"""
Functions to compute the covariance matrix given the data and Kernel Functions
"""

import numpy as np
from GPKernelFunctions import SqExponentialRad #for default kernel function

####################################################################################################
#The following are for multiplicative GPs plus white noise
#     K' = diag(mf) * K * diag(mf) + d**2 #where d are the uncertainties
#     Kss' = diag(mf_ss) * Kss * diag(mf_ss) + dss**2
#     Ks' = diag(mf_ss) * Ks * diag(mf)

def CovarianceMatrixMult(theta,X,KernelFunction,mf,mf_pars,mf_args):
  """
  X - input matrix (n x D)
  theta - hyperparameter array/list
  K - (n x n) covariance matrix
  """
  
  m = mf(mf_pars,mf_args) * np.ones(X.shape[0])
  K = KernelFunction(X,X,theta,white_noise=False)  
  Kp = np.diag(m) * K * np.diag(m) + (np.identity(X.shape[0]) * (theta[-1]**2))
  
  return np.matrix(Kp)

def CovarianceMatrixBlockMult(theta,X,Y,KernelFunction,mf,mf_pars,mf_args_pred,mf_args):
  """
  X - input matrix (q x D) - of training points
  Y - input matrix (n x D) - of precitive points
  theta - hyperparameter array/list
  K - (q x n) covariance matrix block
  """
  
  m = mf(mf_pars,mf_args) * np.ones(Y.shape[0])
  ms = mf(mf_pars,mf_args_pred) * np.ones(X.shape[0])
  K = KernelFunction(X,Y,theta,white_noise=False)
  Kp = np.diag(ms) * K * np.diag(m)
  
  return np.matrix(Kp)

def CovarianceMatrixCornerDiagMult(theta,X,KernelFunction,mf,mf_pars,mf_args_pred,WhiteNoise=True):
  """
  X - input matrix (q x D) - of training points
  theta - hyperparameter array/list
  K - (q x q) covariance matrix corner block - only diagonal terms are returned
    (this function needs optimised as it calculates the whole convariance matrix first...)
  """

  ms = mf(mf_pars,mf_args_pred) * np.ones(X.shape[0])
  K = KernelFunction(X,X,theta,white_noise=False)  
  Kp = np.diag(ms) * K * np.diag(ms)
  if WhiteNoise: Kp += (np.identity(X.shape[0]) * (theta[-1]**2))
  
  Kp = np.diag(np.diag(Kp))
  
  return np.matrix(Kp)

def CovarianceMatrixCornerFullMult(theta,X,KernelFunction,mf,mf_pars,mf_args_pred,WhiteNoise=True):
  """
  X - input matrix (q x D) - of training points
  theta - hyperparameter array/list
  K - (q x q) covariance matrix corner block
  """
  
  ms = mf(mf_pars,mf_args_pred) * np.ones(X.shape[0])
  K = KernelFunction(X,X,theta,white_noise=False)  
  Kp = np.diag(ms) * K * np.diag(ms)
  if WhiteNoise: Kp += (np.identity(X.shape[0]) * (theta[-1]**2))
  
  return np.matrix(Kp)

####################################################################################################
