
import numpy as np
import scipy.linalg as LA

from . import GPCovarianceMatrix as GPC


def GPlogPost(p,x,y,xmf,mf,kf,n_mfp):
  """
  Function to calculate the log likeihood indepentent of class
  May be useful for parallelising MCMCs or testing speed
  
  """

  #calculate the residuals
  r = y - mf(p[:n_mfp],xmf)
    
  #get covariance
  K = kf(x,x,p[n_mfp:])
  
  #get cholesky decomposition and log determinant
  choFactor = LA.cho_factor(K)
  logdetK = (2*np.log(np.diag(choFactor[0])).sum())
  
  #calculate the log likelihood
  logP = -0.5 * np.dot(r.T,LA.cho_solve(choFactor,r)) - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)
  
  return np.float(logP)





