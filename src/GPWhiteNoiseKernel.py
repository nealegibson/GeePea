"""
Useful (non-periodic) GP Kernel Functions
"""

import numpy as np

###################################################################################################
def WhiteNoise(yerr,theta):
  """
  Special kernel for simple white noise models. Returns the diagonal of the covariance
  matrix. GPregression methods will not work for this trival case.

  Parameters
  ----------
  yerr : array of errors for the y data
  theta : noise scale factor beta, where sigma_i = beta * yerr_i or if noise is homoskedastic, sigma_i = beta

  Returns
  -------
  diag(S) : diagonal of the covariance matrix, ie :math:`\sigma_i^2`

  """
  
  #Calculate distance matrix without scaling
  sig = theta[0] * yerr
  
  return sig**2

#add kernel attributes
WhiteNoise.n_par = 1
WhiteNoise.kernel_type = 'White'
WN = WhiteNoise # and alias
###################################################################################################
