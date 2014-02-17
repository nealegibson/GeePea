"""
Useful (non-periodic) GP Kernel Functions
"""

import numpy as np
import scipy.spatial
from scipy.special import gamma,kv

###################################################################################################
#Exponential class
def SqExponentialRad(X,Y,theta,white_noise=False):
  """
  Standard squared exponential function (just one length parameter).

  k(x,x') = th0^2 * exp( - 1/2*th1^2 Sum_i * (x_i-x_i')^2 ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - inverse length scale (1/2l^2)
  theta[2] - white noise standard deviation if white_noise=True
  
  X,Y - input matricies
  """
  
  #Calculate distance matrix without scaling
  D2 = EuclideanDist2(X,Y)
  
  #Calculate covariance matrix
  K = theta[0]**2 * np.exp( - D2 / (2*(theta[1]**2)) )
  
  #Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[2]**2))
  
  return np.matrix(K)

def SqExponentialARD(X,Y,theta,white_noise=False):
  """
  ARD squared exponential function
  (with n inverse length scale for each input in X vectors).
  
  k(x,x') = th0^2 * exp( -Sum_i n_i * (x_i-x_i')^2 ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1:-1] - inverse length scales (1/2l_i^2) for each input vector in X,Y
  theta[-1] - white noise standard deviation if white_noise=True 
  
  X,Y - input matricies
  
  """
  
  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  #n(x_i-x_j)^2 = (sqrt(n)*x_i-sqrt(n)*x_j)^2
  D2 = EuclideanDist2(X,Y,v=np.sqrt(np.abs(np.array(theta[1:-1]))))
  
  #Calculate covariance matrix (leave out the factor of 1/2)
  K = theta[0]**2 * np.exp( -D2 )
  
  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)
  
  return np.matrix(K)

def SqExponential(X,Y,theta,white_noise=False):
  """
  ARD squared exponential function
  (with n length scales for each input in X vectors).
  
  k(x,x') = th0^2 * exp( -Sum_i n_i * (x_i-x_i')^2 ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1:-1] - inverse length scales (1/2l_i^2) for each input vector in X,Y
  theta[-1] - white noise standard deviation if white_noise=True 
  
  X,Y - input matricies
  
  """
  
  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  #n(x_i-x_j)^2 = (sqrt(n)*x_i-sqrt(n)*x_j)^2
  D2 = EuclideanDist2(X,Y,v=1./(np.array(theta[1:-1])*np.sqrt(2.)))
  
  #Calculate covariance matrix (leave out the factor of 1/2)
  K = theta[0]**2 * np.exp( -D2 )
  
  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)
  
  return np.matrix(K)

def ExponentialRad(X,Y,theta,white_noise=False):
  """
  Standard Exponential function (with single length scale).

  k(x,x') = th0^2 * exp( - 1/2*th1^2 Sum_i * (x_i-x_i') ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - inverse length scale (1/2l^2)
  theta[2] - white noise standard deviation if white_noise=True
  
  X,Y - input matricies
  """

  #Calculate distance matrix with scaling
  D = EuclideanDist(X,Y,v=None)

  #Calculate covariance matrix
  K = theta[0]**2 * np.exp( - D / (2*(theta[1]**2)) )

  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)
  
  return np.matrix(K)

def ExponentialARD(X,Y,theta,white_noise=False):
  """
  ARD squared exponential function
  (with n inverse length vector for each input in X vectors).
  
  k(x,x') = th0^2 * exp( -Sum_i n_i * (x_i-x_i') ) [+ sigma^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1:-1] - inverse length scales (1/2l_i^2) for each input vector in X,Y
  theta[-1] - white noise standard deviation if white_noise=True 
  
  X,Y - input matricies
  
  """

  #Calculate distance matrix with scaling
  D = EuclideanDist(X,Y,v=np.sqrt(np.abs(np.array(theta[1:-1]))))

  #Calculate covariance matrix
  K = theta[0]**2 * np.exp( - D / 2 )

  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)
  
  return np.matrix(K)

####################################################################################################
#Rational quadratic - not tested
def RationalQuadRad(X, Y, theta, white_noise = False):
  """
  Rational quadratic kernel (radial) - not fully tested
  
  k(x,x') = th0^2 * (1 + (x_i-x_i')^2/2th1*th2^2)^-th1) [+ th5^2 delta_']
  
  theta[0] - sqrt maximum covariance parameter - gives 1sigma prior dist size
  theta[1] - alpha
  theta[2] - length scale
  theta[3] - white noise standard deviation if white_noise=True

  """

  # Calculate distance matrix without scaling
  D2 = EuclideanDist2(X, Y)

  # Calculate covariance matrix
  K = theta[0]**2 * (1 + (D2 / (2.*theta[1]*(theta[2]**2.)) ) )**(-theta[1])

  # Add white noise
  if white_noise == True: K += (np.identity(X[:,0].size) * (theta[3]**2))

  return np.matrix(K)

####################################################################################################
#Matern class of covariance functions - not tested
def MaternRad(X,Y,theta,white_noise=False):
  """
  Matern covariance kernel - not properly tested!
  Radial - ie same length scales in all inputs
  
  """
  
  #Calculate distance matrix with (global) scaling
  D = EuclideanDist(X,Y) / theta[2]
  
  #Calculate covariance matrix from matern function
  v = theta[1]
  K = 2.**(1.-v) / gamma(v) * (np.sqrt(2.*v)*D)**v * kv(v,np.sqrt(2.*v)*D)
  
  #diagonal terms should be set to one (when D2 = 0, kv diverges but full function = 1)
  #this only works for square 'covariance' matrix...
  #ie fails for blocks..;
#  K[np.where(np.identity(X[:,0].size)==1)] = 1.
  #this should work, but again needs tested properly...
  K[np.where(D==0.)] = 1.

  #now multiply by an overall scale function
  K = K * theta[0]**2
  
  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[3]**2)
  
  return np.matrix(K)

#matern kernel for v=3/2 fixed - rougher than sq exponential
def MAT_Kernel32(X,Y,theta,white_noise=False):
  """
  Matern covariance kernel for 3/2 shape parameter
  
  theta[0] - overall scale param - ie prior covariance
  theta[1] - length scale
  theta[2] - white noise
  
  """
  
  D = EuclideanDist(X,Y) / theta[1] 
  K = theta[0]**2 * (1 + np.sqrt(3.)*D) * np.exp(-np.sqrt(3.)*D)
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[2]**2)
  return np.matrix(K)

#matern kernel for v=5/2 fixed - rougher than sq exponential, smoother than above
#3/2 process
def MAT_Kernel52(X,Y,theta,white_noise=False):
  """
  Matern covariance kernel for 5/2 shape parameter
  
  theta[0] - overall scale param - ie prior covariance
  theta[1] - length scale
  theta[2] - white noise
  
  """
  
  D = EuclideanDist(X,Y) / theta[1] 
  K = theta[0]**2 * (1 + np.sqrt(5.)*D + 5./3.*(D**2)) * np.exp(-np.sqrt(5.)*D)
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[2]**2)
  return np.matrix(K)

def MaternARD(X,Y,theta,white_noise=False):
  """
  Matern covariance kernel - not fully tested!
  different length scales in all inputs
  
  theta[0] - overall scale param - ie prior covariance
  theta[1] - shape parameter
  theta[2:-1] - length scales
  theta[-1] - white noise
  
  """

  #Calculate distance matrix with scaling
  D = EuclideanDist(X,Y,v=theta[2:-1])
  
  #Calculate covariance matrix from matern function
  v = theta[1]
  K = 2**(1.-v) / gamma(v) * (np.sqrt(2*v)*D)**v * kv(v,np.sqrt(2*v)*D)

  #diagonal terms should be set to one (when D2 = 0, kv diverges but full function = 1)
  #this only works for square 'covariance' matrix...
  #ie fails for blocks..;
#  K[np.where(np.identity(X[:,0].size)==1)] = 1.
  #this should work, but again needs tested properly...
  K[np.where(D==0.)] = 1.

  #now multiply by an overall scale function
  K = K * theta[0]
  
  #Add white noise
  if white_noise == True: K += np.identity(X[:,0].size) * (theta[-1]**2)
  
  return np.matrix(K)

####################################################################################################
#Auxilary functions to compute euclidean distances
def EuclideanDist(X1,X2,v=None):
  """
  Calculate the distance matrix for 2 data matricies
  X1 - n x D input matrix
  X2 - m x D input matrix
  v - weight vector
  D - output an n x m matrix of dist = sqrt( Sum_i (1/l_i^2) * (x_i - x'_i)^2 )
  
  """
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v != None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * V
    X2 = X2 * V
  
  #calculate sqaured euclidean distance (after weighting)
  D = scipy.spatial.distance.cdist( X1, X2, 'euclidean')
  
  return D

def EuclideanDist2(X1,X2,v=None):
  """
  Calculate the distance matrix squared for 2 data matricies
  X1 - n x D input matrix
  X2 - m x D input matrix
  v - weight vector
  D2 - output an n x m matrix of dist^2 = Sum_i (1/l_i^2) * (x_i - x'_i)^2
  
  """
  
  #ensure inputs are in matrix form
  X1,X2 = np.matrix(X1), np.matrix(X2)
  
  if v != None: #scale each coord in Xs by the weight vector
    V = np.abs(np.matrix( np.diag(v) ))
    X1 = X1 * V
    X2 = X2 * V
  
  #calculate sqaured euclidean distance (after weighting)
  D2 = scipy.spatial.distance.cdist( X1, X2, 'sqeuclidean' )
  
  return D2

####################################################################################################
