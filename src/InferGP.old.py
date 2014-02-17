
import numpy as np
import scipy.linalg as LA

import GPCovarianceMatrix as GPC
import GPRegression as GPR
import GPUtils as GPU
import GPKernelFunctions as GPK

class GP:
  """
  GP class, updated from the GaussianProcesses model, redesigned to be passable directly
  to Infer's functions, and to allow priors to easily be defined. Also stores the
  cholesky factorisation and log determinant of the covariance matrix, so it is not
  recalculated for identical kernel hyperparameters. This allows the same class to be used
  for ML-typeII and full GP, plus blocked Gibbs sampling, etc...
  
  t - training set/target values
  kf_args - arguments to kernel function, must be a matrix, with columns of input vectors
    (simplest us Nx1 matrix, ie X = np.mat([x1,]).T)
  p - parameters [hyperparameters] + [mean function parameters (if any)]
  kf - kernel function, by default this is SqExponentialARD
  mf - mean function, by default just returns 0.
  n_hp - number of hyperparameters, only needs set with a mean function
  kf_args_pred/mf_args_pred - arguments for predictive distributions, by default set to be
    the same as the input args
  
  """
  
  def __init__(self,t,kf_args,kf=None,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None):
    """
    Initialise the parameters of the GP.
    """
    
    #required arguments
    self.t = t
    self.kf_args = kf_args
    
    #set defaults for optional arguments
    self.pars = np.array([])
    self.mf_args = None
    self.n_hp = None
    self.kf = GPK.SqExponential
    self.kf_args_pred = None
    self.mf_args_pred = None
    
    #keyword arguments
    #calculate covariance matrix, Cholesky decompostion and log determinant if p supplied
    if p != None:
      self.pars = p
      self.K = GPC.CovarianceMatrix(p[:n_hp],self.kf_args,KernelFunction=self.kf)
      self.ChoFactor = LA.cho_factor(self.K)#,overwrite_a=1)
      self.logdetK = (2*np.log(np.diag(self.ChoFactor[0])).sum())
    if kf != None: self.kf = kf
    if mf != None: self.mf = mf
    if mf_args != None: self.mf_args = mf_args
    if kf_args_pred != None: self.kf_args_pred = kf_args_pred
    if mf_args_pred != None: self.mf_args_pred = mf_args_pred
    if n_hp != None: self.n_hp = n_hp
    
    #calculate covariance matrix, Cholesky decompostion and log determinant if p supplied
    if p != None:
      hpar = np.array(p[:self.n_hp])
      mf_par = np.array(p[self.n_hp:])
      self.K = GPC.CovarianceMatrix(hpar,self.kf_args,KernelFunction=self.kf)
      self.ChoFactor = LA.cho_factor(self.K)#,overwrite_a=1)
      self.logdetK = (2*np.log(np.diag(self.ChoFactor[0])).sum())

  def Set(self,t=None,kf_args=None,kf=None,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None):
    """
    Convenience function to reset the parameters of the GP. Pretty much a clone of the
    __init__ method with all the keywords.
    """

    if t != None:self.t = t
    if kf_args != None:self.kf_args = kf_args    
    if p != None:
      self.pars = p   
      self.K = GPC.CovarianceMatrix(p[:n_hp],self.kf_args,KernelFunction=self.kf)
      self.ChoFactor = LA.cho_factor(self.K)#,overwrite_a=1)
      self.logdetK = (2*np.log(np.diag(self.ChoFactor[0])).sum())
    if kf != None: self.kf = kf
    if mf != None: self.mf = mf
    if mf_args != None: self.mf_args = mf_args
    if kf_args_pred != None: self.kf_args_pred = kf_args_pred
    if mf_args_pred != None: self.mf_args_pred = mf_args_pred
    if n_hp != None: self.n_hp = n_hp
  
  def kfVec(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.kf_args.getA()[:,i]
        
  def Describe(self):
    "Print the attributes of the GP object."

    print "--------------------------------------------------------------------------------"
    print "GP attributes:"
    print " Target values t:", self.t.shape, type(self.t)
    print " GP input args X:", self.kf_args.shape, type(self.kf_args)
    print " Log Prior:", self.logPrior
    print " Kernel Function:", self.kf
    print " Hyperparameters:", self.pars[:self.n_hp]
    print " Mean Function:", self.mf
    print " MF args:", np.array(self.mf_args).shape, type(self.mf_args)
    print " MF Parameters:", self.pars[self.n_hp:]
    print " Predictive GP args X:", np.array(self.kf_args_pred).shape, type(self.kf_args_pred)
    print " Predictive mf args:", np.array(self.mf_args_pred).shape, type(self.mf_args_pred)
    print "--------------------------------------------------------------------------------"

  def logLikelihood(self,p):
    "Function to calculate the log likeihood"
    
    #calculate the residuals
    r = self.t - self.mf(p[self.n_hp:],self.mf_args)
    
    #ensure r is an (n x 1) column vector
    r = np.matrix(np.array(r).flatten()).T
    
    #calculate covariance matrix, cholesky factor and logdet if hyperparameters change
#     print "pars:", self.pars, type(self.pars)
#     print "p:", p, type(p)
#     print p[:self.n_hp] != self.pars[:self.n_hp]
#     print np.all(p[:self.n_hp] != self.pars[:self.n_hp])
    if self.pars == None or np.all(p[:self.n_hp] != self.pars[:self.n_hp]):
#       print "no :("
      self.K = GPC.CovarianceMatrix(p[:self.n_hp],self.kf_args,KernelFunction=self.kf)
      self.ChoFactor = LA.cho_factor(self.K)#,overwrite_a=1)
      self.logdetK = (2*np.log(np.diag(self.ChoFactor[0])).sum())
#     else: print "yeah!!"
    
    #store the new parameters
    self.pars = np.copy(p)
    
    #calculate the log likelihood
    logP = -0.5 * r.T * np.mat(LA.cho_solve(self.ChoFactor,r)) - 0.5 * self.logdetK - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)
  
  @staticmethod #ensure static, so it can redefined using a 'normal' function
  def logPrior(p,nhp):
    """
    default log prior, keep hyperparameters > 0
    egs:
    >> object.logPrior = lambda *args: 0.
    >> object.logPrior = lambda p,nhp: -np.inf if (np.array(p[:nhp])<0).any() else 0.
    >> from scipy.stats.distributions import gamma,norm as norm_dist
    >> object.logPrior = lambda p,nhp: np.log(norm_dist.pdf(p[6],.10,0.02)).sum()
    or combine them in a proper function...
    """
    
    #keep all kernel hyperparameters >=0
    return -np.inf if (np.array(p[:nhp])<0).any() else 0.
  
  #define (log) posterior simply as the sum of likelihood and prior
  def logPosterior(self,p):
    "Function to calculate the log posterior"
    
    log_Prior = self.logPrior(p,self.n_hp)
    if log_Prior == -np.inf: return -np.inf
    else: return self.logLikelihood(p) + log_Prior
  
  #default mean function - static method so can be redefined, just returns 0
  @staticmethod
  def mf(*args):
    "default mean function = 0."
    return 0.
  
  def MeanFunction(self):
    "Returns the mean function evaluated at the current parameters"
    return self.mf(self.pars[self.n_hp:], self.mf_args)
  
  def mfRes(self):
    "Returns the residuals from the mean function"
    return self.t - self.mf(self.pars[self.n_hp:], self.mf_args)
  
  def GPRes(self):
    "Return residuals from the GP + mf"
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrix(self.pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self.pars[:self.n_hp],self.kf_args,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerDiag(self.pars[:self.n_hp],self.kf_args,KernelFunction=self.kf,WhiteNoise=wn)
    
    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )
    
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self.pars[self.n_hp:], self.mf_args) #subtract the mean function
    
    #and do the regression on the residuals...
    t_pred, t_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)
    
    return self.t - self.mf(self.pars[self.n_hp:], self.mf_args) - t_pred
  
  def PredictGP(self,X_pred=None,mf_args_pred=None,wn=True):
    "Returns the predictive distributions for the GP alone using current hyperparmeters"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrix(self.pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self.pars[:self.n_hp],self.kf_args_pred,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerDiag(self.pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )
    
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self.pars[self.n_hp:], self.mf_args) #subtract the mean function
    
    #and do the regression on the residuals...
    t_pred, t_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)
    
#     #...and add the mean function back
#     t_pred += self.mf(self.pars[self.n_hp:], self.mf_args_pred)
    
    #return the predictive mean and variance
    return t_pred, t_pred_err
  
  def Predict(self,X_pred=None,mf_args_pred=None,wn=True):
    """
    Returns the predictive distributions for the GP and current hyperparmeters plus the
    mean function
    """
    
    #get the predictive distributions for the GP alone
    t_pred, t_pred_err = self.PredictGP(X_pred=X_pred,mf_args_pred=mf_args_pred,wn=wn)
        
    #...and add the mean function back
    t_pred += self.mf(self.pars[self.n_hp:], self.mf_args_pred)
    
    #return the predictive mean and variance
    return t_pred, t_pred_err
  
  def GetRandomVector(self,wn=False):
    "Returns a random vector from the conditioned GP"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrix(self.pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self.pars[:self.n_hp],self.kf_args_pred,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerFull(self.pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
    #get precision matrix
    PrecMatrix = np.linalg.inv( np.matrix(K) )
              
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self.pars[self.n_hp:],self.mf_args) #subtract the mean function
    
    #predictive mean
    m = self.mf(self.pars[self.n_hp:], self.mf_args_pred)
        
    return GPU.RandVectorFromConditionedGP(K_s,PrecMatrix,K_ss,r,m=m)
  
  def GetRandomVectorFromPrior(self,wn=False):
    "Returns a random vector from the GP prior"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args

    #Construct the covariance matrix
    K_ss = GPC.CovarianceMatrixCornerFull(self.pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
    return GPU.RandomVector(K_ss) + self.mf(self.pars[self.n_hp:], self.mf_args_pred)

