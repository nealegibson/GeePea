
import numpy as np
import scipy.linalg as LA
import pylab

import GPCovarianceMatrix as GPC
import GPRegression as GPR
import GPUtils as GPU
import GPKernelFunctions as GPK
import Optimiser as OP

#import ToeplitzSolve
import GPToeplitz as GPT

def add_n_par(N):
  """
  Simple decorator function to add n_par to a static function
  """
  def decor(func):
    func.n_par = N
    return func
  return decor

class GP(object):
  # """
  # GP class, updated from the Infer model, redesigned to be passable directly
  # to InferTools functions, and to allow priors to easily be defined. Also stores the
  # cholesky factorisation and log determinant of the covariance matrix, so it is not
  # recalculated for identical kernel hyperparameters. This allows the same class to be used
  # for ML-typeII and full GP, plus blocked Gibbs sampling, etc...
  #
  # t - training set/target values
  # kf_args - arguments to kernel function, must be a matrix, with columns of input vectors
  #   (simplest is Nx1 matrix, ie X = np.mat([x1,]).T)
  # p - parameters [hyperparameters] + [mean function parameters (if any)]
  # kf - kernel function, by default this is SqExponentialARD
  # mf - mean function, by default just returns 0., but needs to be in the format mf(pars,mf_args)
  # mf_args - arguments to mean function - usually a 1D vector
  # n_hp - number of hyperparameters, only needs set with a mean function
  # kf_args_pred/mf_args_pred - arguments for predictive distributions, by default set to be
  #   the same as the input args
  # n_store - number of cholesky factors etc to be remembered. For a blocked Gibbs MCMC, with
  #   acc ~25%, 4-5 should be enough to avoid recalculating chol + logdetK. 1 is of course
  #   enough for a ML-typeII MCMC, where the first inversion will be calculated only if the
  #   hyperparameters do not vary. The hyperparameters are stored as hashes, but this doesn't
  #   make a noticeable diff to speed for only a few hp. The self.si keyword is indexed every
  #   time a new set of hp are used, and cycles around n_store before overwriting older
  #   calculations. A few hundred is probably ok (but unnecessary) for smallish matrices
  #
  # """
  
  # def __init__(self,t,kf_args,kf=GPK.SqExponential,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None,fp=None,n_store=1,toeplitz_kf=None):
  #   """
  #   Initialise the parameters of the GP.
  #   """
  #
  #   #required arguments
  #   self.t = t
  #   self.kf_args = kf_args
  #   #convert kf_args to matrix if only one input is provided
  #   if self.kf_args.ndim == 1:
  #     self.kf_args = np.mat(self.kf_args,).T
  #
  #   #set defaults for optional arguments
  #   self._pars = np.array([])
  #   self.mf_args = mf_args
  #   self.n_hp = n_hp
  #   self.kf = kf
  #   if mf is not None: self.mf = mf
  #   self.kf_args_pred = kf_args_pred
  #   #convert kf_args to matrix if only one input is provided
  #   if kf_args_pred is not None:
  #     if self.kf_args_pred.ndim == 1:
  #       self.kf_args_pred = np.mat(self.kf_args_pred,).T
  #   self.mf_args_pred = mf_args_pred
  #   self.fp = fp
  #
  #   #set the toplitz likelihood if the toeplitz kernel is defined
  #   self.toeplitz_kf = toeplitz_kf
  #   if self.toeplitz_kf is not None:
  #     self.teop_sol = np.mat(np.empty(t.size)).T
  #     self.logLikelihood = self.logLikelihood_toeplitz
  #
  #   #storage parameter defaults
  #   self.n_store = n_store
  #   self.hp_hash = np.empty(n_store)
  #   self.ChoFactor = [[] for q in range(n_store)]
  #   self.logdetK = [[] for q in range(n_store)]
  #
  #   #keyword arguments
  #   self.si = 0 #index used to store the chol solve, etc
  #   self.n_hp = n_hp
  #   if p is not None: self.Pars(p) #always set _pars via the Pars method, p is the visible property
  
  def __init__(self,x,y,p=None,kf=GPK.SqExponential,n_hp=None,n_mfp=None,kernel_type='Full'\
    ,x_pred=None,mf=None,xmf=None,xmf_pred=None,n_store=1,ep=None,fp=None,logPrior=None):
    """
    Initialise the GP. See class docstring for a description of the inputs.
    
    inputs
    ------
    x - arguments to kernel function, should be an NxD array/matrix with aribtrary no of
      input vectors. Can also be a 1D array, which is converted to Nx1 matrix
    y - training set/target values
    p - parameters of the GP. list/array of mf parameters followed by GP hyperparameters
    kf - kernel function, by default this is SqExponential, see GPKernelFunctions for
      more details
    n_hp - number of hyperparamters of the kernel, only needs specified if mf is provided
      This can also be obtained via an attribute of the kernel function, ie kf.n_par
      There are multiple ways of defining this via classes, simple function attributes, etc
      This can also be a function that accepts the dimensions of x, ie kf.n_par(D)
    n_mfp - number of mean function parameters, only needs specified if mf is provided
      Like n_hp, this can also be obtained via an attribute of the mean function, ie mf.n_par
      There are multiple ways of defining this via classes, simple function attributes, etc
      This can also be a function that accepts the dimensions of x, ie mf.n_par(D), although
      I'm not sure why you'd ever want to do that
    *Note only one of n_hp and n_mfp needs to be set, either directly in the class definition
      or via function attributes. The built in kernels already have correct attributes set,
      and they are easy to create for user defined kernels. (see GPKernelFunctions.py for egs)
    k_type - kernel type, defines what type of solver is used. By default is 'Full' (Cholesky)
      solver. Other options are Toeplitz/T and White/W
    mf - mean function, by default just returns 0., but needs to be in the format mf(pars,mf_args)
    xmf - arguments to the mf. If identical to x do not need to be specified again
    x_pred - predictive arguments to kernel function, same format as x, requires same no
      of inputs D, but N can vary
    xmf_pred - predictive arguments for mean function. If identical to x_pred (or x), do
      not need to be specified
    n_store - number of cholesky factors etc to be remembered. For a blocked Gibbs MCMC, with
      acc ~25%, 4-5 should be enough to avoid recalculating chol + logdetK. 1 is of course
      enough for a ML-typeII MCMC, where the first inversion will be calculated only if the
      hyperparameters do not vary. The hyperparameters are stored as hashes, but this doesn't
      make a noticeable diff to speed for only a few hp. The self.si keyword is indexed every
      time a new set of hp are used, and cycles around n_store before overwriting older
      calculations. A few hundred is probably ok (but unnecessary) for smallish matrices.
      This is only used for the cholesky solver for now.
    ep - list/array of parameter error estimates corresponding to input parameters p
    fp - list/array of fixed parameters corresponding to p, required to fix parameters for
      optimisation. 0 means vary, 1 = fixed. If ep is specified then it is taken directly
      from it
    logPrior - logPrior (static) function of the form logPrior(p,n_hp)
      by default returns -np.inf if (np.array(p[hyper])<0).any() else 0., ie all hyperparams must be >=0
    
    """

    #initialise some optional variables
    self.fp = fp
    self.ep = ep
    self.n_hp = n_hp
    self.n_mfp = n_mfp
    self.xmf = xmf
    self.x_pred = x_pred
    self.xmf_pred = xmf_pred
    self.si = 0
    self.n_store = n_store
    self.hp_hash = np.empty(n_store)
    self.ChoFactor = [[] for q in range(n_store)]
    self.logdetK = [[] for q in range(n_store)]
    self._pars = np.array([])

    #pass arguments to set_pars function to propertly initialise everything
    self.set_pars(x=x,y=y,p=p,kf=GPK.SqExponential,n_hp=n_hp,n_mfp=n_mfp,kernel_type=kernel_type\
      ,x_pred=x_pred,mf=mf,xmf=xmf,xmf_pred=xmf_pred,n_store=n_store,ep=ep,fp=fp,logPrior=logPrior)

  def set_pars(self,x=None,y=None,p=None,kf=None,n_hp=None,n_mfp=None,kernel_type=None\
    ,x_pred=None,mf=None,xmf=None,xmf_pred=None,n_store=None,ep=None,fp=None,logPrior=None):
    """
    Set the parameters of the GP. See class docstring for a description of the inputs.
    """

    #GP parameters
    if y is not None:
      self.y = y
      self.N = y.size
    if x is not None:
      self.x = np.mat(x).reshape(self.N,-1) # set x as matrix, and reshape so N x D
      self.D = self.x.shape[1] # set x as matrix, and reshape so N x D
    if p is not None:
      self.Pars(p)
    if kf is not None:
      self.kf = kf
      if kernel_type is None:
        try: kernel_type = kf.kernel_type
        except: pass
    if kernel_type is not None:
      self.kernel_type = kernel_type
      #set likelihood from kernel type
      if self.kernel_type == 'Full':
        self.logLikelihood = self.logLikelihood_cholesky
      if self.kernel_type == 'Toep' or self.kernel_type == 'T':
        self.logLikelihood = self.logLikelihood_toeplitz
        self.teop_sol = np.mat(np.empty(self.N)).T
      if self.kernel_type == 'White' or self.kernel_type == 'W':
        self.logLikelihood = self.logLikelihood_white
    if x_pred is not None:
      self.x_pred = x_pred
    if logPrior is not None:
      self.logPrior = logPrior
    
    #mean function parameters
    if n_hp is not None: self.n_hp = n_hp
    if n_mfp is not None: self.n_mfp = n_mfp
    if mf is not None:
      self.mf = mf
    if xmf is not None: self.xmf = xmf
    if xmf_pred is not None: self.xmf = xmf_pred
    
    #auxiliary parameters
    if n_store is not None:
      self.n_store = n_store
      self.hp_hash = np.empty(self.n_store)
      self.ChoFactor = [[] for q in range(self.n_store)]
      self.logdetK = [[] for q in range(self.n_store)]
    if ep is not None: self.ep = np.array(ep)
    if fp is not None: self.fp = np.array(fp)
    
    #set fixed parameteres if ep set but not fp
    if self.fp is None and self.ep is not None:
      self.fp = (self.ep > 0.) * 1
    
    #try and set the number of mf/hyperparameters via mf/kf attributes/func if not defined directly
    self.n_par = self.p.size
    if self.n_mfp is None:
      try: self.n_mfp = self.mf.n_par(self.D)
      except:
        print "mf.n_par() function not found!"
        try: self.n_mfp = self.mf.n_par
        except: print "mf.n_par parameter not found!"
        else: print "#mf par set from mf.n_par!"
      else: print "#mf par set from mf.n_par(D)!"
    if self.n_hp is None:
      try: self.n_hp = self.kf.n_par(self.D)
      except:
        print "mf.n_par() function not found!"
        try: self.n_hp = self.kf.n_par
        except: print "kf.n_par parameter not found!"
        else: print "#kf par set from kf.n_par!"
      else: print "#kf par set from kf.n_par(D)!"

    #set number of parameters for kernel and mean function
    if self.n_mfp is not None and self.n_hp is None:
      self.n_hp = self.n_par - self.n_mfp
    if self.n_hp is not None and self.n_mfp is None:
      self.n_mfp = self.n_par - self.n_hp
    if self.n_hp is None and self.n_mfp is None:
      self.n_mfp = 0
      self.n_hp = self.n_par - self.n_mfp

    #print warning if mf is set without knowing the number of parameters
    if mf is not None and self.n_mfp is None:
      print "warning: mean function was changed but n_mfp is not set!"
      print "please set n_hp or n_mfp via set_pars method...!"

    #set kf function args and predictive args
    if self.x_pred is None: self.x_pred = self.x #set x_pred to x if pred not given
    #set mf function args and predictive args
    if self.n_mfp > 0: #only need set if mean function changed
      if self.xmf is None: #set mean function args
        self.xmf = self.kfVec() #set as first vector of x by default
        self.xmf_pred = self.kfVec_pred() #set as first vector of x by default
      else:
        if self.xmf_pred is None: self.xmf_pred = self.xmf

#   def Set(self,t=None,kf_args=None,kf=None,p=None,mf=None,mf_args=None,n_hp=None,kf_args_pred=None,mf_args_pred=None,fp=None):
#     """
#     Convenience function to reset the parameters of the GP. Pretty much a clone of the
#     __init__ method with all the keywords.
#     """
# 
#     if t is not None:self.t = t
#     if kf_args is not None:self.kf_args = kf_args    
#     
#     #keyword arguments
#     self.si = 0 #index used to store the chol solve, etc
#     if n_hp is not None: self.n_hp = n_hp
#     if p is not None: self.Pars(p)
#     if kf is not None: self.kf = kf
#     if mf is not None: self.mf = mf
#     if fp is not None: self.fp = fp
#     if mf_args is not None: self.mf_args = mf_args
#     if kf_args_pred is not None: self.kf_args_pred = kf_args_pred
#     if mf_args_pred is not None: self.mf_args_pred = mf_args_pred
  
  def Pars(self,p=None):
    """
    Simple function to return or set pars. Required as _pars is semi-private, and does not
    compute cho factor if set directly, eg MyGP._pars = [blah], plus should be a np.array.
    
    """
    
    if p is None:
      return np.copy(self._pars)
    else:
      # reset the hash thingy is pars are reset
      self._pars = np.array(p)
      self.hp_hash[self.si] = hash('') #create empty hash that won't be matched with any pars
      self.ChoFactor[self.si] = None
      self.logdetK[self.si] = None
  
  """create p as a property, ie can set p directly, eg MyGP.p = [blah...], and Pars will be called"""
  p = property(Pars,Pars)

  #add n_hp and n_mfp as properties, as they should be calculated from each other...
  def set_n_hp(selfself,n_hp=None):
    if n_hp is None:
      return np.copy(self.n_hp)
    else:
      self.n_hp = n_hp
      self.n_mfp = self.n_par - self.n_hp

  def set_n_mfp(selfself,n_mfp=None):
    if n_mfp is None:
      return np.copy(self.n_mfp)
    else:
      self.n_mfp = n_mfp
      self.n_hp = self.n_par - self.n_mfp

  n_hp = property(set_n_hp,set_n_hp)
  n_mfp = property(set_n_mfp,set_n_mfp)

  def mfPars(self):
    return self._pars[self.n_hp:]
  
  def kfVec(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.x.getA()[:,i]

  def kfVec_pred(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.x_pred.getA()[:,i]

  def describe(self):
    "Print the attributes of the GP object."

    print "--------------------------------------------------------------------------------"
    print "GP attributes:"
    print " Target values y:", self.y.shape, type(self.y)
    print " GP input args x:", self.x.shape, type(self.x)
    print " Log Prior:", self.logPrior
    print " Kernel Function:", self.kf.__name__
    print " Kernel Type:", self.kernel_type
    print " Hyperparameters:", self._pars[:self.n_hp], "(#hp = {})".format(self.n_hp)
    print " Mean Function:", self.mf.__name__
    print " MF args:", np.array(self.xmf).shape, type(self.xmf)
    print " MF Parameters:", self._pars[self.n_hp:], "(#mfp = {})".format(self.n_mfp)
    print " Predictive GP args X:", np.array(self.x_pred).shape, type(self.x_pred)
    print " Predictive mf args:", np.array(self.xmf_pred).shape, type(self.xmf_pred)
    print "--------------------------------------------------------------------------------"

  def logLikelihood_cholesky(self,p):
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
      self.ChoFactor[self.si] = LA.cho_factor(GPC.CovarianceMatrix(p[:self.n_hp],self.kf_args,KernelFunction=self.kf))
      self.logdetK[self.si] = (2*np.log(np.diag(self.ChoFactor[self.si][0])).sum())
      self.hp_hash[self.si] = new_hash
    
    #calculate the log likelihood
    logP = -0.5 * r.T * np.mat(LA.cho_solve(self.ChoFactor[useK],r)) - 0.5 * self.logdetK[useK] - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)
  
  def logLikelihood_toeplitz(self,p):
    """
    Function to calculate the log likeihood using a Toeplitz matrix
    Need to define a separate Toeplitz kernel
    """
    
    #calculate the residuals
    r = self.t - self.mf(p[self.n_hp:],self.mf_args)
    
    #ensure r is an (n x 1) column vector
    r = np.matrix(np.array(r).flatten()).T
    
    #calculate the log likelihood using the Toplitz kernel
    #logP = -0.5 * r.T * np.mat(LA.cho_solve(self.ChoFactor[useK],r)) - 0.5 * self.logdetK[useK] - (r.size/2.) * np.log(2*np.pi)
    #make sure white noise is set to true!
#    logdetK,self.teop_sol = ToeplitzSolve.LTZSolve(self.toeplitz_kf(self.kf_args,p[:self.n_hp],white_noise=True),r)
    logdetK,self.teop_sol = GPT.LTZSolve(GPT.CovarianceMatrixToeplitz(p[:self.n_hp],self.kf_args,self.toeplitz_kf),r,self.teop_sol)
    logP = -0.5 * r.T * self.teop_sol - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)

  def logLikelihood_white(self,p):
    "Function to calculate the log likeihood"
    pass
    # #calculate the residuals
    # r = self.t - self.mf(p[self.n_hp:],self.mf_args)
    #
    # #ensure r is an (n x 1) column vector
    # r = np.matrix(np.array(r).flatten()).T
    #
    # #check if covariance, chol factor and log det are already calculated and stored
    # new_hash = hash(p[:self.n_hp].tostring()) # calculate and check the hash
    # if np.any(self.hp_hash == new_hash):
    #   useK = np.where(self.hp_hash == new_hash)[0][0]
    # else: #else calculate and store the new hash, cho_factor and logdetK
    #   useK = self.si = (self.si+1) % self.n_store #increment the store index number
    #   self.ChoFactor[self.si] = LA.cho_factor(GPC.CovarianceMatrix(p[:self.n_hp],self.kf_args,KernelFunction=self.kf))
    #   self.logdetK[self.si] = (2*np.log(np.diag(self.ChoFactor[self.si][0])).sum())
    #   self.hp_hash[self.si] = new_hash
    #
    # #calculate the log likelihood
    # logP = -0.5 * r.T * np.mat(LA.cho_solve(self.ChoFactor[useK],r)) - 0.5 * self.logdetK[useK] - (r.size/2.) * np.log(2*np.pi)
    #
    # return np.float(logP)
  
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
  #add_n_par just adds a function attribute to specify number of mf parameters
  @staticmethod
  @add_n_par(0)
  def mf(*args):
    "default mean function = 0."
    return 0.

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
    K = GPC.CovarianceMatrix(self._pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self._pars[:self.n_hp],self.kf_args,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerDiag(self._pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    
    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )
    
    #need do the regression on the *residual data* if mean funciton exists
    r = self.t - self.mf(self._pars[self.n_hp:], self.mf_args) #subtract the mean function
    
    #and do the regression on the residuals...
    t_pred, t_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)
    
    return self.t - self.mf(self._pars[self.n_hp:], self.mf_args) - t_pred
  
  def PredictGP(self,X_pred=None,mf_args_pred=None,wn=True):
    "Returns the predictive distributions for the GP alone using current hyperparmeters"
    
    if X_pred is not None: self.kf_args_pred = X_pred
    if mf_args_pred is not None: self.mf_args_pred = mf_args_pred
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred is None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrix(self._pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self._pars[:self.n_hp],self.kf_args_pred,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerDiag(self._pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
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
    if fp is not None: self.fp = fp
    pars = OP.Optimise(self.logPosterior,self._pars,(),fixed=self.fp,method='NM',**kwargs)
    self.Pars(pars)
  
  def GetRandomVector(self,wn=False):
    "Returns a random vector from the conditioned GP"
    
    #set predictive distributions to the inputs if not set
    if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred is None: self.mf_args_pred = self.mf_args
    
    #Construct the covariance matrix
    K = GPC.CovarianceMatrix(self._pars[:self.n_hp],self.kf_args,KernelFunction=self.kf)
    K_s = GPC.CovarianceMatrixBlock(self._pars[:self.n_hp],self.kf_args_pred,self.kf_args,KernelFunction=self.kf)
    K_ss = GPC.CovarianceMatrixCornerFull(self._pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
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
    if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
    if self.mf_args_pred is None: self.mf_args_pred = self.mf_args

    #Construct the covariance matrix
    K_ss = GPC.CovarianceMatrixCornerFull(self._pars[:self.n_hp],self.kf_args_pred,KernelFunction=self.kf,WhiteNoise=wn)
    
    return GPU.RandomVector(K_ss) + self.mf(self._pars[self.n_hp:], self.mf_args_pred)
  
  def PlotRanges(self,x=None,wn=True,**kwargs):
    """Plots the 1 and 2 sigma range of the GP (but doesn't take into account mean function errors)"""
    
    if x==None:
      if self.mf_args_pred is None:
        if self.mf_args is not None: self.mf_args_pred = self.mf_args      
        else:
          if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
          self.mf_args_pred = self.kfVec_pred()
    else:
      self.mf_args_pred = x
    
    t_pred,t_pred_err = self.Predict(wn=wn)
    GPU.PlotRanges(self.mf_args_pred,t_pred,t_pred_err,**kwargs)
    
  def PlotData(self,**kwargs):
    """Plots the data with errorbars"""
        
    if self.mf_args is None: self.mf_args = self.kfVec()
    
    #set the errors    
    if self.n_hp is None: err = self._pars[-1]
    else: err = self._pars[self.n_hp-1]
    
    GPU.PlotData(self.mf_args,self.t,np.ones(self.t.size)*err,title=None,**kwargs)
  
  def PlotMean(self,x=None):
    
    if x==None:
      if self.mf_args_pred is None:
        if self.mf_args is not None: self.mf_args_pred = self.mf_args      
        else:
          if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
          self.mf_args_pred = self.kfVec_pred()
    else:
      self.mf_args_pred = x
    
    pylab.plot(self.mf_args_pred,self.MeanFunctionPred()*np.ones(self.mf_args_pred.size),'r--')
  
  def Plot(self):
    """Convenience method to call both PlotRanges and PlotData with defaults"""
    
    self.PlotRanges()
    self.PlotData()
    self.PlotMean()
    
    
    
  