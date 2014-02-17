"""
Useful (periodic) GP Kernel Functions

This module contains some GP kernel functions written by Chris Hart
in 2011, modified by Neale Gibson and Suzanne Aigrain.

These include periodic and quasi-periodic kernel functions, some with gradient
calculations which can be used to speed up optimisation (not yet incorporated into
the GPClass due - ideally the kernel and kernel gradient need to be written as a
class with a common inverse covariance matrix to avoid repeating calculations, which
will be hopefully incorporated in future versions)

****
ALL THE PERIODIC AND QUASI-PERIODIC KERNELS USE FREQUENCY, RATHER
THAN PERIOD, AS A HYPERPARAMETER
****

"""

import numpy as np
import scipy.spatial
from GaussianProcesses.GPKernelFunctions import EuclideanDist, EuclideanDist2

def PeriodicSqExponentialRad(X, Y, theta, white_noise = False):
  """
  Periodic squared exponential kernel function (similar to Eq 4.31 in
  Rasmussen & Williams 2006). Hyperparameters are amplitude, frequency,
  length scale, and white noise variance.
  (only tested for 1D inputs)

  k(x,x') = th0^2 * exp( - 2 * sin^2 [th1*pi*(x_i-x_i')] / th2^2 ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - frequency
  theta[2] - length scale
  theta[3] - white noise standard deviation if white_noise=True
  
  """

  # Calculate distance matrix without scaling
  D = EuclideanDist(X,Y)

  # Scale to make it periodic  
  Dscaled = np.pi * D * theta[1]

  # Calculate covariance matrix
  K = theta[0]**2 * np.exp( - 2*np.sin(Dscaled)**2 / (theta[2]**2) )

  # Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[3]**2))

  return np.matrix(K)

def QuasiPeriodicSqExponentialRad(X, Y, theta, white_noise = False, grad_output=False):
  """
  Quasi-periodic squared exponential function, used to describe
  quasi-periodic behaviour with a single evolutionary
  time-scale. Constructed by combining the Periodic and
  SquaredExponentialRad kernels (just multiplied together).
  Hyperparameters are amplitude frequency, periodic length scale,
  non-periodic length scale, and white noise variance.
  (only tested for 1D inputs)
  
  k(x,x') = th0^2 * exp( - sin^2 [th1*pi*(x_i-x_i')] / 2*th2^2 )* exp( - (x_i-x_i')^2 / 2*th3^2) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - frequency
  theta[2] - length scale of periodic term
  theta[3] - length scale of multiplicative sq exp
  theta[4] - white noise standard deviation if white_noise=True
  
  """
  # Calculate distance matrix without scaling
  D = EuclideanDist(X,Y)

  # Scale to make it periodic  
  Dscaled = np.pi * D * theta[1]  

  # Calculate covariance matrix
  K = theta[0]**2 * np.exp( - np.sin(Dscaled)**2 / (2 * (theta[2]**2)) - ( D**2 / (2*theta[3]**2) ) )

  # Calculate gradients
  d0 = 2 * K / theta[0]
  d1 = - Dscaled * np.sin(Dscaled) * np.cos(Dscaled) * K / (theta[1] * theta[2]**2)
  d2 = np.sin(Dscaled)**2 * K / theta[2]**3
  d3 = D**2 * K / theta[3]**3
  d4 = 2 * (np.identity(X[:,0].size) * (theta[4]))

  # Add white noise
  if white_noise == True: 
    K += (np.identity(X[:,0].size) * (theta[4]**2))

  if grad_output == True:
    return np.matrix(K), np.matrix(d0), np.matrix(d1), np.matrix(d2), np.matrix(d3), np.matrix(d4)
  return np.matrix(K)  

def QuasiPeriodicRationalQuadRad(X, Y, theta, white_noise = False):
  """
  Quasi-periodic rational quadratic function, used to describe
  quasi-periodic behaviour with a range of evolutionary
  time-scales. Constructed by combining the Periodic and
  RationalQuadratic kernels.  Hyperparameters are amplitude,
  frequency, periodic length scale, non-periodic length scale,
  rational quadratic index, and white noise variance.
  
  (RQ: k(x,x') = (1 + d^2/2alpha*l^2)^-alpha)
  k(x,x') = th0^2 * exp( - sin^2 [th1*pi*(x_i-x_i')] / 2*th2^2 )* (1 + (x_i-x_i')^2/2*th3*th4^2)^(-th3) [+ th5^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - frequency
  theta[2] - periodic length scale
  theta[3] - alpha of RQ
  theta[4] - l of RQ
  theta[5] - white noise standard deviation if white_noise=True
  
  """
  
  # Calculate distance matrix without scaling
  D = EuclideanDist(X, Y)
  D2 = EuclideanDist2(X, Y)

  # Scale to make it periodic  
  Dscaled = np.pi * D * theta[1]  

  # Calculate covariance matrix
  K = theta[0]**2 * np.exp( -np.sin(Dscaled)**2 / (2.*theta[2]**2.)) * (1 + (D2 / (2.*theta[3]*theta[4]**2) ) )**(-theta[3])

  # Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[5]**2))

  return np.matrix(K)

def QuasiPeriodicSqExponentialRadPlus(X, Y, theta, white_noise = True, grad_output=False):
  """
  Quasi-periodic squared exponential function, but with additive
  squared exponential term to describe (e.g.) correlated noise or
  short-term variability.  Hyperparameters are amplitude, frequency,
  periodic length scale, non-periodic length scale, additive term
  amplitude, additive term length scale, and white noise variance.

  ie same as QuasiPeriodicSqExponentialRad with the addition of another
  independent Sq exponential term
  
  k(x,x') = th0^2 * exp( - sin^2 [th1*pi*(x_i-x_i')] / 2*th2^2 )* exp( - (x_i-x_i')^2 / 2*th3^2)
               + th[4]**2 * np.exp( - D**2 / (2*th[5]**2)) [+ sigma^2 delta_']

  (for periodic kernel)
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - frequency
  theta[2] - length scale of periodic term
  theta[3] - length scale of multiplicative sq exp
  (for additive sq exp)
  theta[4] - sqrt max covariance
  theta[5] - length scale of additive sq exp
  (white noise)
  theta[6] - white noise standard deviation if white_noise=True
  """

  # Calculate distance matrix without scaling
  D = EuclideanDist(X,Y)

  # Scale to make it periodic  
  Dscaled = np.pi * D * theta[1]  

  # Calculate covariance matrix
  Ka = theta[0]**2 * np.exp( - np.sin(Dscaled)**2 / (2 * (theta[2]**2)) - ( D**2 / (2*theta[3]**2) ) )
  Kb = theta[4]**2 * np.exp( - D**2 / (2*theta[5]**2))
  K = Ka + Kb  
  
  d0 = 2 * Ka / theta[0]
  d1 = - Dscaled * np.sin(Dscaled) * np.cos(Dscaled) * Ka / (theta[1] * theta[2]**2)
  d2 = np.sin(Dscaled)**2 * Ka / theta[2]**3
  d3 = D**2 * Ka / theta[3]**3
  d4 = 2 * Kb / theta[4]
  d5 = D**2 * Kb / theta[5]**3 
  d6 = 2 * (np.identity(X[:,0].size) * (theta[6]))

  # Add white noise
  if white_noise == True: 
    K += (np.identity(X[:,0].size) * (theta[6]**2))

  if grad_output == True:
    return np.matrix(K), np.matrix(d0), np.matrix(d1), np.matrix(d2), np.matrix(d3), np.matrix(d4), np.matrix(d5), np.matrix(d6)
  return np.matrix(K) 


