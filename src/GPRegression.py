"""
GP Regression Model
"""
import numpy as np
import time

####################################################################################################

def GPRegress(K_s,PrecMatrix,K_ss,y,return_covariance=False):
  """
  Given input array of kernel values, precision matrix, etc, get predictive
  distribution using Gaussian Process Regression.
  (may be room for improvement as I only need to calculate diagonal elements
  of covariance matrix for predictive points)

  INPUTS:
  K_s - (q x n) covariance function of predictive points relative to test points
  PrecMatrix - (n x n) precision matrix of training points
  K_ss - (q x q) Covariance matrix of predictive points
  y - (n x 1) y values of training points
  
  added an option to return the full covariance matrix, if needed to compare
  regression on different inputs (added for spot models)
  
  """
  
  #ensure all data are in matrix form
  K_s = np.matrix(K_s)
  K_ss = np.matrix(K_ss)
  PrecMatrix = np.matrix(PrecMatrix)
  y = np.matrix(np.array(y)).T # (n x 1) column vector
  
  # (q x n) = (q x n) * (n x n) * (n x 1)
  f_s = K_s * PrecMatrix * y
  
  # (q x q) = (q x q) - (q x n) * (n x n) * (n x q)  
  var_s = K_ss - np.matrix(K_s) * PrecMatrix * np.matrix(K_s).T

  #return predictive values and stddev for each input vector
  if return_covariance: return np.array(f_s).flatten(), np.array(var_s)
  else: return np.array(f_s).flatten(), np.array(np.sqrt(np.diag(var_s)))

####################################################################################################
