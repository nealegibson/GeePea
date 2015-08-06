
import numpy as np
import scipy.linalg as LA
import pylab

import MGPCovarianceMatrix as GPC
import GPRegression as GPR
import GPUtils as GPU
import GPKernelFunctions as GPK
import Optimiser as OP

class MGP(object):
  """
  Multiplicative GP class, updated from the GaussianProcesses model, redesigned to be passable
  directly to Infer's functions, and to allow priors to easily be defined. Also stores the
  cholesky factorisation and log determinant of the covariance matrix, so it is not
  recalculated for identical kernel hyperparameters. This allows the same class to be used
  for ML-typeII and full GP, plus blocked Gibbs sampling, etc...
  
  Note this is a multiplicative GP, using an affine transform of a multivariate normal.
  Just look at the wikipedia article for more details, thanks to T. Evans for pointing this
  out!
    x ~ N(mu,Sig)
    if y = c + B * mu
    y ~ N(c+B*mu,B*Sig*B.T)
    
    c is just a standard addition to the GP (as used in the additive GP), therefore
    doesn't change the covariance matrix
    
    In this case, B is a diagonal matrix containing the mean function, but it can represent
    any transform
    
    It is trivial to calculate the covariance matrix blocks
    K' = diag(mf) * K * diag(mf)
    Kss' = diag(mf_ss) * Kss * diag(mf_ss) - ie predictive mean function
    The blocks are a little trickier:
    Ks' = diag(mf_ss) * Ks * diag(mf) - to produce an M x N block
    
    This is understood from the following matrix partition, where B is the transformation
    of the data, and Bss the transformation of the predictive points:
    
                | B*k*B.T      B*ks.T*Bss.T |
    B*Sig*B.T = |                           |
                | Bss*ks*B.T  Bss*kss*Bss.T |
    
  Note that the white noise is added to the diagonal after the affine transform!, ie
    K' = diag(mf) * K * diag(mf) + d**2 #where d are the uncertainties
    Kss' = diag(mf_ss) * Kss * diag(mf_ss) + dss**2
    Ks' = diag(mf_ss) * Ks * diag(mf)
  
  The plots have also been updated, and the default mean function is 1, although the diff
  between additive and multiplicative GPs is irrelevant without a mean function! 
  
  t - training set/target values
  kf_args - arguments to kernel function, must be a matrix, with columns of input vectors
    (simplest us Nx1 matrix, ie X = np.mat([x1,]).T)
  p - parameters [hyperparameters] + [mean function parameters (if any)]
  kf - kernel function, by default this is SqExponentialARD
  mf - mean function, by default just returns 1., but needs to be in the format mf(pars,mf_args)
  mf_args - arguments to mean function - usually a 1D vector
  n_hp - number of hyperparameters, only needs set with a mean function
  kf_args_pred/mf_args_pred - arguments for predictive distributions, by default set to be
    the same as the input args
  n_store - number of cholesky factors etc to be remembered. For a blocked Gibbs MCMC, with
    acc ~25%, 4-5 should be enough to avoid recalculating chol + logdetK. 1 is of course
    enough for a ML-typeII MCMC, where the first inversion will be calculated only if the
    hyperparameters do not vary. The hyperparameters are stored as hashes, but this doesn't
    make a noticeable diff to speed for only a few hp. The self.si keyword is indexed every
    time a new set of hp are used, and cycles around n_store before overwriting older
    calculations. A few hundred is probably ok (but unnecessary) for smallish matrices
  
  """
  
  def __init__(self,t,kf_args,kf=GPK.SqExponential,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None,fp=None,n_store=1):
    """
    Initialise the parameters of the GP.
    """
    
    print "\n  *** important - MGP is experimental and may give weird results ***\n"
    
    #required arguments
    self.t = t
    self.kf_args = kf_args
    #convert kf_args to matrix if only one input is provided
    if self.kf_args.ndim == 1:
      self.kf_args = np.mat(self.kf_args,).T
    
    #set defaults for optional arguments
    self._pars = np.array([])
    self.mf_args = mf_args
    self.n_hp = n_hp
    self.kf = kf
    if mf != None: self.mf = mf
    self.kf_args_pred = kf_args_pred
    #convert kf_args to matrix if only one input is provided
    if kf_args_pred is not None:
      if self.kf_args_pred.ndim == 1:
        self.kf_args_pred = np.mat(self.kf_args_pred,).T
    self.mf_args_pred = mf_args_pred
    self.fp = fp

    #storage parameter defaults
    self.n_store = n_store
    self.hp_hash = np.empty(n_store)
    self.ChoFactor = [[] for q in range(n_store)]
    self.logdetK = [[] for q in range(n_store)]
    
    #keyword arguments
    self.si = 0 #index used to store the chol solve, etc
    self.n_hp = n_hp
    if p != None: self.Pars(p) #always set _pars via the Pars method, p is the visible property
    
  def Set(self,t=None,kf_args=None,kf=None,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None,fp=None):
    """
    Convenience function to reset the parameters of the GP. Pretty much a clone of the
    __init__ method with all the keywords.
    """

    if t != None:self.t = t
    if kf_args != None:self.kf_args = kf_args    
    
    #keyword arguments
    self.si = 0 #index used to store the chol solve, etc
    if n_hp != None: self.n_hp = n_hp
    if p != None: self.Pars(p)
    if kf != None: self.kf = kf
    if mf != None: self.mf = mf
    if fp != None: self.fp = fp
    if mf_args != None: self.mf_args = mf_args
    if kf_args_pred != None: self.kf_args_pred = kf_args_pred
    if mf_args_pred != None: self.mf_args_pred = mf_args_pred
  
  def Pars(self,p=None):
    """
    Simple function to return or set pars. Required as _pars is semi-private, and does not
    compute cho factor if set directly, eg MyGP._pars = [blah], plus should be a np.array.
    
    """
    
    if p == None:
      return np.copy(self._pars)
    else:
      self._pars = np.array(p)
      self.hp_hash[self.si] = hash(np.array(p[:self.n_hp]).tostring())
      self.ChoFactor[self.si] = LA.cho_factor(GPC.CovarianceMatrixMult(self._pars[:self.n_hp],self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args))
      self.logdetK[self.si] = 2*np.log(np.diag(self.ChoFactor[0][0])).sum()
  
  """create p as a property, ie can set p directly, eg MyGP.p = [blah...], and Pars will be called"""
  p = property(Pars,Pars)
  
  def mfPars(self):
    return self._pars[self.n_hp:]
  
  def kfVec(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.kf_args.getA()[:,i]

  def kfVec_pred(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.kf_args_pred.getA()[:,i]
        
  def Describe(self):
    "Print the attributes of the GP object."

    print "--------------------------------------------------------------------------------"
    print "GP attributes:"
    print " Target values t:", self.t.shape, type(self.t)
    print " GP input args X:", self.kf_args.shape, type(self.kf_args)
    print " Log Prior:", self.logPrior
    print " Kernel Function:", self.kf.__name__
    print " Hyperparameters:", self._pars[:self.n_hp]
    print " Mean Function:", self.mf.__name__
    print " MF args:", np.array(self.mf_args).shape, type(self.mf_args)
    print " MF Parameters:", self._pars[self.n_hp:] if self.n_hp else "none"
    print " Predictive GP args X:", np.array(self.kf_args_pred).shape, type(self.kf_args_pred)
    print " Predictive mf args:", np.array(self.mf_args_pred).shape, type(self.mf_args_pred)
    print "--------------------------------------------------------------------------------"
  
  def logLikelihood(self,p):
    "Function to calculate the log likeihood"
        
    #calculate the residuals
    r = self.t - self.mf(p[self.n_hp:],self.mf_args)
    
    #ensure r is an (n x 1) column vector
    r = np.matrix(np.array(r).flatten()).T
        
    #check if covariance, chol factor and log det are already calculated and stored
    new_hash = hash(p[:self.n_hp].tostring()) # calculate and check the hash
    if np.any(self.hp_hash == new_hash): 
      useK = np.where(self.hp_hash == new_hash)[0][0]
    else: #else calculate and store the new hash, cho_factor and logdetK
      useK = self.si = (self.si+1) % self.n_store #increment the store index number
      self.ChoFactor[self.si] = LA.cho_factor(GPC.CovarianceMatrixMult(p[:self.n_hp],self.kf_args,self.kf,self.mf,p[self.n_hp:],self.mf_args))
      self.logdetK[self.si] = (2*np.log(np.diag(self.ChoFactor[self.si][0])).sum())
      self.hp_hash[self.si] = new_hash
    
    #calculate the log likelihood
    logP = -0.5 * r.T * np.mat(LA.cho_solve(self.ChoFactor[useK],r)) - 0.5 * self.logdetK[useK] - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)
  
  @staticmethod #ensure static, so it can redefined using a 'normal' function
  def logPrior(p,nhp):
    """
    default log prior, keep hyperparameters > 0
    should be of the form logp(p,n_hp) where n_hp is number of hyperparameters
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
    return 1.
  
  def MeanFunction(self):
    "Returns the mean function evaluated at the current parameters"
    return self.mf(self._pars[self.n_hp:], self.mf_args)

  def MeanFunctionPred(self):
    "Returns the mean function evaluated at the current parameters"
    return self.mf(self._pars[self.n_hp:], self.mf_args_pred)
  
  def mfRes(self):
    "Returns the residuals from the mean function"
    return self.t - self.mf(self._pars[self.n_hp:], self.mf_args)
  
  def GPRes(self):
    "Return residuals from the GP + mf"
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrixMult(self._pars[:self.n_hp],self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args)
    K_s = GPC.CovarianceMatrixBlockMult(self._pars[:self.n_hp],self.kf_args,self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args,self.mf_args)
    K_ss = GPC.CovarianceMatrixCornerDiagMult(self._pars[:self.n_hp],self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args)
    
    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )
    
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self._pars[self.n_hp:], self.mf_args) #subtract the mean function
    
    #and do the regression on the residuals...
    t_pred, t_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)
    
    return self.t - self.mf(self._pars[self.n_hp:], self.mf_args) - t_pred
  
  def PredictGP(self,X_pred=None,mf_args_pred=None,wn=True):
    "Returns the predictive distributions for the GP alone using current hyperparmeters"
    
    if X_pred != None: self.kf_args_pred = X_pred
    if mf_args_pred != None: self.mf_args_pred = mf_args_pred

    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrixMult(self._pars[:self.n_hp],self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args)
    K_s = GPC.CovarianceMatrixBlockMult(self._pars[:self.n_hp],self.kf_args_pred,self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args_pred,self.mf_args)
    K_ss = GPC.CovarianceMatrixCornerDiagMult(self._pars[:self.n_hp],self.kf_args_pred,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args_pred,WhiteNoise=wn)
        
    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )
    
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self._pars[self.n_hp:], self.mf_args) #subtract the mean function
    
    #and do the regression on the residuals...
    t_pred, t_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)
    
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
    t_pred += self.mf(self._pars[self.n_hp:], self.mf_args_pred)
    
    #return the predictive mean and variance
    return t_pred, t_pred_err
  
  def Optimise(self,method='NM',fp=None,**kwargs):
    """
    Optimise the parameters of the model - simple wrapper to Infer.Optimise
    """
    
    print "Guess pars:", self._pars
    if fp != None: self.fp = fp
    pars = OP.Optimise(self.logPosterior,self._pars,(),fixed=self.fp,method='NM',**kwargs)
    self.Pars(pars)
  
  def GetRandomVector(self,wn=False):
    "Returns a random vector from the conditioned GP"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrixMult(self._pars[:self.n_hp],self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args)
    K_s = GPC.CovarianceMatrixBlockMult(self._pars[:self.n_hp],self.kf_args_pred,self.kf_args,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args_pred,self.mf_args)
    K_ss = GPC.CovarianceMatrixCornerDiagMult(self._pars[:self.n_hp],self.kf_args_pred,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args_pred,WhiteNoise=wn)
    
    #get precision matrix
    PrecMatrix = np.linalg.inv( np.matrix(K) )
              
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self._pars[self.n_hp:],self.mf_args) #subtract the mean function
    
    #predictive mean
    m = self.mf(self._pars[self.n_hp:], self.mf_args_pred)
    
    return GPU.RandVectorFromConditionedGP(K_s,PrecMatrix,K_ss,r,m=m)
  
  def GetRandomVectorFromPrior(self,wn=False):
    "Returns a random vector from the GP prior"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred == None: self.mf_args_pred = self.mf_args

    #Construct the covariance matrix
    K_ss = GPC.CovarianceMatrixCornerDiagMult(self._pars[:self.n_hp],self.kf_args_pred,self.kf,self.mf,self._pars[self.n_hp:],self.mf_args_pred,WhiteNoise=wn)
    
    return GPU.RandomVector(K_ss) + self.mf(self._pars[self.n_hp:], self.mf_args_pred)
  
  def PlotRanges(self,x=None,wn=True,**kwargs):
    """Plots the 1 and 2 sigma range of the GP (but doesn't take into account mean function errors)"""
    
    if x==None:
      if self.mf_args_pred == None:
        if self.mf_args != None: self.mf_args_pred = self.mf_args      
        else:
          if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
          self.mf_args_pred = self.kfVec_pred()
    else:
      self.mf_args_pred = x
    
    t_pred,t_pred_err = self.Predict(wn=wn)
    GPU.PlotRanges(self.mf_args_pred,t_pred,t_pred_err,**kwargs)
    
  def PlotData(self,**kwargs):
    """Plots the data with errorbars"""
    
    if self.mf_args == None: self.mf_args = self.kfVec()
    
    #set the errors    
    if self.n_hp == None: err = self._pars[-1]
    else: err = self._pars[self.n_hp-1]
    
    GPU.PlotData(self.mf_args,self.t,np.ones(self.t.size)*err,title=None,**kwargs)
  
  def PlotMean(self,x=None):
    
    if x==None:
      if self.mf_args_pred == None:
        if self.mf_args != None: self.mf_args_pred = self.mf_args      
        else:
          if self.kf_args_pred == None: self.kf_args_pred = self.kf_args
          self.mf_args_pred = self.kfVec_pred()
    else:
      self.mf_args_pred = x

    pylab.plot(self.mf_args_pred,self.MeanFunctionPred()*np.ones(self.mf_args_pred.size),'r--')
  
  def Plot(self):
    """Convenience method to call both PlotRanges and PlotData with defaults"""
    
    self.PlotRanges()
    self.PlotData()
    self.PlotMean()
    
