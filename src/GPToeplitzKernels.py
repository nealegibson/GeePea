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
  r"""
  Toeplitz squared exponential function - only 1D input therefore 3 accepts 3 parameters. See
  GeePea.SqExponential for equivalent full kernel.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi^2 exp\left( - \sum_{k=1}^K \eta_k (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi,l,\sigma\}`, :math:`\X = \{x_1,\dots,x_n \}^T`,
  and :math:`\Y = \{y_1,\dots,y_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x 1 matrix of inputs
  Y : N' x 1 matrix of inputs
  theta : array of 3 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  SqExponential : Squared exponential kernel using standard length scales

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

def ToeplitzMatern32(X,Y,theta,white_noise=False):
  r"""
  Toeplitz Matern 3/2 function - only 1D input therefore accepts 3 parameters. See
  GeePea.Matern32 for equivalent full kernel.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\sqrt(3)D\right) exp\left( -\sqrt(3)D \right) + \delta_{ij}\sigma^2,

  where :math:`D = (x_{i} - x_{k}) / l`, :math:`\th = \{\xi,l,\sigma\}`,
  :math:`\X = \{x_1,\dots,x_n \}^T`, and :math:`\Y = \{y_1,\dots,y_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x 1 matrix of inputs
  Y : N' x 1 matrix of inputs
  theta : array of 3 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  Matern32 : Full Matern 3/2 kernel

  """

  #first calculate distance matrix
  D = np.array(X-Y[0]).flatten() / theta[1]

  #calculate Toeplitz 'matrix' stored as array
  a = theta[0]**2 * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)

  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a  

ToeplitzMatern32.n_par = 3
ToeplitzMatern32.kernel_type = 'Toeplitz'

##########################################################################################
def ToeplitzMatern32_inv(X,Y,theta,white_noise=False):
  r"""
  Toeplitz Matern 3/2 function using inverse length scale - only 1D input therefore
  accepts 3 parameters. See GeePea.Matern32 for equivalent full kernel.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\sqrt(3)D\right) exp\left( -\sqrt(3)D \right) + \delta_{ij}\sigma^2,

  where :math:`D = (x_{i} - x_{k}) * \eta`, :math:`\th = \{\xi,\eta,\sigma\}`,
  :math:`\X = \{x_1,\dots,x_n \}^T`, and :math:`\Y = \{y_1,\dots,y_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x 1 matrix of inputs
  Y : N' x 1 matrix of inputs
  theta : array of 3 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  Matern32 : Full Matern 3/2 kernel

  """

  #first calculate distance matrix
  D = np.array(X-Y[0]).flatten() * theta[1]

  #calculate Toeplitz 'matrix' stored as array
  a = theta[0]**2 * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)

  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a  

ToeplitzMatern32_inv.n_par = 3
ToeplitzMatern32_inv.kernel_type = 'Toeplitz'

##########################################################################################

def ToeplitzMatern32_inv_log(X,Y,theta,white_noise=False):
  r"""
  Toeplitz Matern 3/2 function using inverse length scale - only 1D input therefore
  accepts 3 parameters. See GeePea.Matern32 for equivalent full kernel.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\sqrt(3)D\right) exp\left( -\sqrt(3)D \right) + \delta_{ij}\sigma^2,

  where :math:`D = (x_{i} - x_{k}) * \eta`, :math:`\th = \{\xi,\eta,\sigma\}`,
  :math:`\X = \{x_1,\dots,x_n \}^T`, and :math:`\Y = \{y_1,\dots,y_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x 1 matrix of inputs
  Y : N' x 1 matrix of inputs
  theta : array of 3 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  Matern32 : Full Matern 3/2 kernel

  """

  #first calculate distance matrix
  D = np.array(X-Y[0]).flatten() * np.exp(theta[1])

  #calculate Toeplitz 'matrix' stored as array
  a = np.exp(2*theta[0]) * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)

  #add white noise
  if white_noise == True: a[0] += (theta[-1]**2)
  return a  

ToeplitzMatern32_inv_log.n_par = 3
ToeplitzMatern32_inv_log.kernel_type = 'Toeplitz'

##########################################################################################









