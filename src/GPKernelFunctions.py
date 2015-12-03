
import numpy as np
import scipy.spatial
from scipy.special import gamma,kv

###################################################################################################
def SqExponential(X,Y,theta,white_noise=False):
  r"""
  Squared exponential kernel function (with length scale for each K inputs in X/Y matrices).

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi^2 exp\left( - \sum_{k=1}^K \frac{(x_{ik} - x_{jk})^2}{2l_k^2} ) \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi,l_1\dots l_k,\sigma\}`, :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  SqExponentialARD : Squared exponential kernel using inverse length scales

  """

  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  #n(x_i-x_j)^2 = (sqrt(n)*x_i-sqrt(n)*x_j)^2
  D2 = EuclideanDist2(X,Y,v=1./(np.array(theta[1:-1])))

  #Calculate covariance matrix
  K = theta[0]**2 * np.exp( - 0.5 * D2 )

  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)

  return np.matrix(K)
#add some attributes
SqExponential.n_par = lambda D: D+2
SqExponential.kernel_type = "Full"

###################################################################################################
def SqExponentialARD(X,Y,theta,white_noise=False):
  r"""
  Squared exponential function using inverse length scales for each input dimension. ARD refers to
  *Automatic Relevance Determination*. This is a useful parameterisation for applying shrinkage
  to the inverse length scales, ie if :math:`\eta_k\lim 0` then that input is not relevant to the
  inference. This kernel was used in Gibson et al. (2012) to account for multiple kernel inputs
  using NICMOS data.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi^2 exp\left( - \sum_{k=1}^K \eta_k (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi,\eta_1\dots\eta_k,\sigma\}`, :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  SqExponential : Squared exponential kernel using standard length scales

  """

  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  #n(x_i-x_j)^2 = (sqrt(n)*x_i-sqrt(n)*x_j)^2
  D2 = EuclideanDist2(X,Y,v=np.sqrt(np.abs(np.array(theta[1:-1]))))

  #Calculate covariance matrix (leave out the factor of 1/2)
  K = theta[0]**2 * np.exp( -D2 )

  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)

  return np.matrix(K)
#add some attributes
SqExponentialARD.n_par = lambda D: D+2
SqExponentialARD.kernel_type = "Full"

###################################################################################################
def SqExponentialARDLog(X,Y,theta,white_noise=False):
  r"""
  
  Same as SqExponentialARD, only using log inputs
  
  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    e^{2\xi} \exp\left( - \sum_{k=1}^K e^{\eta_k} (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi,\eta_1\dots\eta_k,\sigma\}`, :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  SqExponential : Squared exponential kernel using standard length scales

  """

  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  #n(x_i-x_j)^2 = (sqrt(n)*x_i-sqrt(n)*x_j)^2
  D2 = EuclideanDist2(X,Y,v=np.sqrt(np.exp(np.array(theta[1:-1]))))
  
  #Calculate covariance matrix (leave out the factor of 1/2)
  K = np.exp(2.*theta[0]-D2)
  
  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)

  return np.matrix(K)
#add some attributes
SqExponentialARDLog.n_par = lambda D: D+2
SqExponentialARDLog.kernel_type = "Full"

###################################################################################################
def SqExponentialRad(X,Y,theta,white_noise=False):
  r"""
  Radial squared exponential kernel function, ie. using single length scale for any no of inputs K.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi^2 exp\left( - \frac{1}{2l^2} (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi,l,\sigma\}`, :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of 3 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  See Also
  --------
  SqExponential : Squared exponential kernel using standard length scales for all K inputs
  SqExponentialARD : Squared exponential kernel using inverse length scales for all K inputs

  """
  
  #Calculate distance matrix without scaling
  D2 = EuclideanDist2(X,Y)
  
  #Calculate covariance matrix
  K = theta[0]**2 * np.exp( - D2 / (2.*(theta[1]**2)) )
  
  #Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[2]**2))
  
  return np.matrix(K)
#add some attributes
SqExponentialRad.n_par = 3
SqExponentialRad.kernel_type = "Full"

###################################################################################################
def SqExponentialSum(X,Y,theta,white_noise=False):
  r"""
  Squared exponential function using independent basis components, ie the GP is the sum of
  independent stochastic components. Individual height scale and length scale for each of K inputs.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \sum_{k=1}^K \xi_i^2 exp\left( - \eta_k (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi_1,\eta_1,\dots,\xi_k,\eta_k,\sigma\}`,
  :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of 2*K+1 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  """
  
  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  m,n = X.shape
  #ensure inputs are matrices - otherwise EuclideanDist fails for 1D
  # assert type(X) is np.matrixlib.defmatrix.matrix
  # assert type(Y) is np.matrixlib.defmatrix.matrix

  D2 = EuclideanDist2( np.mat(X[:,0]),np.mat(Y[:,0]),v=[np.sqrt(np.abs(theta[1]))])
  K = theta[0]**2 * np.exp( -D2 )

  # add for remaining inputs
  for i in range(1,n):
    D2 = EuclideanDist2( np.mat(X[:,i]),np.mat(Y[:,i]),v=[np.sqrt(np.abs(theta[2*i+1]))])
    K += theta[2*i]**2 * np.exp( -D2 )

  #Add white noise
  if white_noise == True: K += np.identity(m) * (theta[-1]**2)
  
  return np.matrix(K)
#add some attributes
SqExponentialSum.n_par = lambda D: 2*D+1
SqExponentialSum.kernel_type = "Full"

###################################################################################################
def SqExponentialSumLog(X,Y,theta,white_noise=False):
  r"""
  Same as SqExponentialSumLog, only using log inputs
  
  Squared exponential function using independent basis components, ie the GP is the sum of
  independent stochastic components. Individual height scale and length scale for each of K inputs.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \sum_{k=1}^K e^{2\xi} exp\left( - e^{\eta_k} (x_{ik} - x_{jk})^2 \right) + \delta_{ij}\sigma^2,

  where :math:`\th = \{\xi_1,\eta_1,\dots,\xi_k,\eta_k,\sigma\}`,
  :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of 2*K+1 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  """
  
  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  m,n = X.shape
  #ensure inputs are matrices - otherwise EuclideanDist fails for 1D
  # assert type(X) is np.matrixlib.defmatrix.matrix
  # assert type(Y) is np.matrixlib.defmatrix.matrix

  D2 = EuclideanDist2( np.mat(X[:,0]),np.mat(Y[:,0]),v=[np.sqrt(np.exp(theta[1]))])
#  K = theta[2*i]**2 * np.exp( -D2 )
  K = np.exp( 2.*theta[0]-D2 )

  # add for remaining inputs
  for i in range(1,n):
    D2 = EuclideanDist2( np.mat(X[:,i]),np.mat(Y[:,i]),v=[np.sqrt(np.exp(theta[2*i+1]))])
#    K += theta[2*i]**2 * np.exp( -D2 )
    K += np.exp( 2.*theta[2*i]-D2 )

  #Add white noise
  if white_noise == True: K += np.identity(m) * (theta[-1]**2)
  
  return np.matrix(K)
#add some attributes
SqExponentialSumLog.n_par = lambda D: 2*D+1
SqExponentialSumLog.kernel_type = "Full"

###################################################################################################
def Matern32(X,Y,theta,white_noise=False):
  r"""
  Matern 3/2 kernel function with length scale parameters for each input dimension.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\sqrt(3)D\right) exp\left( -\sqrt(3)D \right) + \delta_{ij}\sigma^2,

  where :math:`D = \sqrt{\sum_{k=1}^K \frac{1}{l_k^2} (x_{ik} - x_{jk})^2}`,
  :math:`\th = \{\xi_1,l_1,\dots,l_k,\sigma\}`,
  :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  """

  #calculate distance matrix, D = sqrt(sum 1/l_i^2 (x_i - x'_i)^2 )
  D = EuclideanDist(X,Y,v=1./(np.array(theta[1:-1])))

  # calculate covariance matrix
  K = theta[0]**2 * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)

  #add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[2]**2)

  return np.matrix(K)
#add attributes
Matern32.n_par = lambda D: D+2
Matern32.kernel_type = "Full"

###################################################################################################
def Matern52(X,Y,theta,white_noise=False):
  r"""
  Matern 5/2 kernel function with length scale parameters for each input dimension.

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\sqrt(5)D + \frac{5}{3}D^2\right) exp\left( -\sqrt(5)D \right) + \delta_{ij}\sigma^2,

  where :math:`D = \sqrt{\sum_{k=1}^K \frac{1}{l_k^2} (x_{ik} - x_{jk})^2}`,
  :math:`\th = \{\xi_1,l_1,\dots,l_k,\sigma\}`,
  :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  """

  #calculate distance matrix, D = sqrt(sum 1/l_i^2 (x_i - x'_i)^2 )
  D = EuclideanDist(X,Y,v=1./(np.array(theta[1:-1])))

  # calculate covariance matrix
  K = theta[0]**2 * (1 + np.sqrt(5.)*D + 5./3.*(D**2)) * np.exp(-np.sqrt(5.)*D)

  #add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[2]**2)

  return np.matrix(K)
#add attributes
Matern52.n_par = lambda D: D+2
Matern52.kernel_type = "Full"

###################################################################################################

#Rational quadratic - not tested
def RationalQuadRad(X, Y, theta, white_noise = False):
  r"""
  Rational quadratic kernel (radial) - not fully tested

  .. math::

    \Bsig_{ij} = k(\bx_i,\bx_j,\th) =
    \xi_i^2 \left(1+\frac{D^2}{2\alpha l^2}\right)^{-\alpha} + \delta_{ij}\sigma^2,

  where :math:`D = \sqrt{\sum_{k=1}^K (x_{ik} - x_{jk})^2}`,
  :math:`\th = \{\xi,\alpha,l,\sigma\}`,
  :math:`\X = \{\bx_1,\dots,\bx_n \}^T`,
  and :math:`\Y = \{\by_1,\dots,\by_{n^\prime}\}^T`.

  Parameters
  ----------
  X : N x K matrix of inputs
  Y : N' x K matrix of inputs
  theta : array of K+2 kernel function parameters
  white_noise : boolean, add white noise to diagonal if True

  Returns
  -------
  K : N x N' covariance matrix

  """

  # Calculate distance matrix without scaling
  D2 = EuclideanDist2(X, Y)
  
  # Calculate covariance matrix
  K = theta[0]**2 * (1 + (D2 / (2.*theta[1]*(theta[2]**2.)) ) )**(-theta[1])

  # Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[3]**2))

  return np.matrix(K)
#add attributes
RationalQuadRad.n_par = 3
RationalQuadRad.kernel_type = "Full"

####################################################################################################
#Auxilary functions to compute euclidean distances
def EuclideanDist(X1,X2,v=None):
  r"""
  Calculate the distance matrix for 2 data matrices

  .. math::

    D = \sqrt{\sum_{k=1}^K v_i^2 (x_i - x^\prime_i)^2}

  Parameters
  ----------
  X1 : N x K matrix of inputs
  X2 : N' x K matrix of inputs
  v : array of weights for each input dimension

  Returns
  -------
  D : N x N' matrix of distance measure

  """
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v is not None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * V
    X2 = X2 * V
  
  #calculate sqaured euclidean distance (after weighting)
  D = scipy.spatial.distance.cdist( X1, X2, 'euclidean')
  
  return D

####################################################################################################
def EuclideanDist2(X1,X2,v=None):
  """
  Calculate the distance matrix squared for 2 data matrices

  .. math::

    D = \sum_{k=1}^K v_i^2 (x_i - x^\prime_i)^2

  Parameters
  ----------
  X1 : N x K matrix of inputs
  X2 : N' x K matrix of inputs
  v : array of weights for each input dimension

  Returns
  -------
  D : N x N' matrix of distance measure

  """
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v is not None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * V
    X2 = X2 * V
  
  #calculate sqaured euclidean distance (after weighting)
  D2 = scipy.spatial.distance.cdist( X1, X2, 'sqeuclidean' )
  
  return D2

####################################################################################################
