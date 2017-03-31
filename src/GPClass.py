
import numpy as np
import scipy.linalg as LA
import pylab

try:
  import dill
  dill_available = 'yes'
except ImportError: dill_available = 'no'
  
import GPCovarianceMatrix as GPC
import GPMultCovarianceMatrix as GPMC
import GPRegression as GPR
import GPUtils as GPU
import GPKernelFunctions as GPK
import Optimiser as OP
import DifferentialEvolution as DE

#import ToeplitzSolve
import GPToeplitz as GPT

###############################################################################################################

class GP(object):
  """
  GP class, updated from the Infer model, redesigned to be passable directly
  to InferTools functions, and to allow priors to easily be defined. Also stores the
  cholesky factorisation and log determinant of the covariance matrix, so it is not
  recalculated for identical kernel hyperparameters. This allows the same class to be used
  for ML-typeII and full GP, plus blocked Gibbs sampling, etc.
  
  :inputs
  -------
  
  x - arguments to kernel function, should be an NxD array/matrix with arbitrary no of
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
    The built in mean function (that returns 0) also has n_par = 0
  k_type - kernel type, defines what type of solver is used. By default is 'Full' (Cholesky)
    solver. Other options are Toeplitz/T and White/W
  gp_type = 'add' - gp type, default is normal, additive gp.
    optional multiplicative gp by setting to 'mult' - note that this is experimental
    mult GPs are based on an affine transform of the GP, suggested by T. Evans.
    Typially they don't change inference of light curve parameters, but often worth checking,
    particularly for large systematics
        x ~ N(mu,Sig)
        if y = c + B * mu
        y ~ N(c+B*mu,B*Sig*B.T)
    and the kernels take the form:
      K' = diag(mf) * K * diag(mf) + d**2 #where d are the uncertainties
      Kss' = diag(mf_ss) * Kss * diag(mf_ss) + dss**2
      Ks' = diag(mf_ss) * Ks * diag(mf)
    Note that the white noise is added to the diagonal after the affine transform!
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
  
  def __init__(self,x,y,p=None,kf=GPK.SqExponential,n_hp=None,n_mfp=None,kernel_type='Full',gp_type='add',
    x_pred=None,mf=None,xmf=None,xmf_pred=None,n_store=1,ep=None,fp=None,logPrior=None,yerr=None,opt=False):
    """
    Initialise the GP. See class docstring for a description of the inputs.
    
    See Class docstring for list of inputs.

    """

    #initialise some optional variables
    self.fp = fp
    self.ep = ep
    self._n_hp = n_hp
    self._n_mfp = n_mfp
    self.xmf = xmf
    self.x_pred = x_pred
    self.xmf_pred = xmf_pred
    self.si = 0
    self.n_store = n_store
    self.hp_hash = np.empty(n_store)
    self.choFactor = [[] for q in range(n_store)]
    self.logdetK = [[] for q in range(n_store)]
    self._pars = np.array([])
    self.yerr = yerr

    #pass arguments to set_pars function to propertly initialise everything
    self.set_pars(x=x,y=y,p=p,kf=kf,n_hp=n_hp,n_mfp=n_mfp,kernel_type=kernel_type,gp_type=gp_type,
      x_pred=x_pred,mf=mf,xmf=xmf,xmf_pred=xmf_pred,n_store=n_store,ep=ep,fp=fp,logPrior=logPrior,yerr=yerr)
    
    #run optimiser?
    if opt: self.opt()
    
  def set_pars(self,x=None,y=None,p=None,kf=None,n_hp=None,n_mfp=None,kernel_type=None,
    x_pred=None,mf=None,xmf=None,xmf_pred=None,n_store=None,ep=None,fp=None,logPrior=None,yerr=None,gp_type=None):
    """
    Set the parameters of the GP. See class docstring for a description of the inputs.

    """
    
    #GP parameters
    if y is not None:
      self.y = np.array(y)
      self.n = self.y.size
    if x is not None:
      self.x = np.array(x)
      if self.x.ndim == 1: #if array is 1D
        self.x = self.x.reshape(-1,1) #reshape so N x 1
      self.x = np.mat(self.x) #finally ensure its a matrix
      assert self.x.shape[0] == self.n,\
        "x is not the correct shape, should be NxD, leading dimension: {} != {}".format(self.x.shape[0],self.n)
      self.d = self.x.shape[1] # record the dimensionality of x
    if x_pred is not None:
      self.x_pred = np.array(x_pred)
      if self.x_pred.ndim == 1: #if array is 1D
        self.x_pred = self.x_pred.reshape(-1,1) #reshape so N x 1
      self.x_pred = np.mat(self.x_pred) #finally ensure its a matrix
      assert self.x_pred.shape[1] == self.d, "x_pred is not the correct shape, should be N_predxD, trailing dimension != D"
    if p is not None:
      self.pars(p)
    if kf is not None:
      self.kf = kf
      try:
        kernel_type = kf.kernel_type #overwrite kernel type if given by kernel
        # print "overwriting default kernel type"
      except: pass
    
    #set kernel type
    if kernel_type is not None:
      self.kernel_type = kernel_type
      #set likelihood from kernel type
      if self.kernel_type == 'Full':
        self.logLikelihood = self.logLikelihood_cholesky
      if self.kernel_type == 'Toeplitz' or self.kernel_type == 'T':
        self.logLikelihood = self.logLikelihood_toeplitz
        self.teop_sol = np.mat(np.empty(self.n)).T
      if self.kernel_type == 'White' or self.kernel_type == 'W':
        self.logLikelihood = self.logLikelihood_white
        if self.yerr is None:
          self.yerr = np.ones(self.n)
    if logPrior is not None:
      self.logPrior = logPrior
    
    #set covariance matrix functions depending on the gp_type    
    if gp_type is not None:
      self.gp_type = gp_type
      if gp_type == 'add': #additive gp
        if self.kernel_type == 'Full': #normal kernel type
          self.CovMat_p = self.CovarianceMatrixAdd_p #cov matrix for likelihood calculations
          self.CovMat = self.CovarianceMatrixFullAdd #full cov matrix of training data
          self.CovMatBlock = self.CovarianceMatrixBlockAdd #cov matrix block K_s - ie training data vs pred points
          self.CovMatCorner = self.CovarianceMatrixCornerAdd #cov matrix corner K_ss - pred points with themselves
          self.CovMatCornerDiag = self.CovarianceMatrixCornerDiagAdd #diagonal of cov matrix corner
        if self.kernel_type == 'Toeplitz': #toeplitz kernel
          self.CovMat_p = self.CovarianceMatrixToeplitzAdd_p #cov matrix for likelihood calculations - only returns vector for Toe
          self.CovMat = self.CovarianceMatrixFullToeplitzAdd
          self.CovMatBlock = self.CovarianceMatrixBlockToeplitzAdd
          self.CovMatCorner = self.CovarianceMatrixCornerToeplitzAdd
          self.CovMatCornerDiag = self.CovarianceMatrixCornerDiagToeplitzAdd
      #need to add support for multiplicative gp kernels
      elif gp_type == 'mult':
        print "############################################################"
        print "# warning: Support for multiplicative GPs is experimental. #"
        print "############################################################"
        if self.kernel_type == 'Full':
          self.CovMat_p = self.CovarianceMatrixMult_p #cov matrix for likelihood calculations
          self.CovMat = self.CovarianceMatrixFullMult #full cov matrix of training data
          self.CovMatBlock = self.CovarianceMatrixBlockMult #cov matrix block K_s - ie training data vs pred points
          self.CovMatCorner = self.CovarianceMatrixCornerMult #cov matrix corner K_ss - pred points with themselves
          self.CovMatCornerDiag = self.CovarianceMatrixCornerDiagMult #diagonal of cov matrix corner
        if self.kernel_type == 'Toeplitz':
          #use same cov matrix for likelihood calculation, toeplitz likelihood is modified for this
          self.CovMat_p = self.CovarianceMatrixToeplitzAdd_p #cov matrix for likelihood calculations - only returns vector for Toe
          #new functions to return full cov matrices after affine transform for toeplitz/multiplicative
          self.CovMat = self.CovarianceMatrixFullToeplitzMult
          self.CovMatBlock = self.CovarianceMatrixBlockToeplitzMult
          self.CovMatCorner = self.CovarianceMatrixCornerToeplitzMult
          self.CovMatCornerDiag = self.CovarianceMatrixCornerDiagToeplitzMult
          #raise ValueError("gp_type '{}' not yet supported for Toeplitz matrices!")
      else:
        raise ValueError("gp_type '{}' is not recognised!")
    
    #mean function parameters
    if n_hp is not None: self._n_hp = n_hp
    if n_mfp is not None: self._n_mfp = n_mfp
    if mf is not None:
      self.mf = mf
    if xmf is not None: self.xmf = np.array(xmf)
    if xmf_pred is not None: self.xmf_pred = np.array(xmf_pred)

    #auxiliary parameters
    if n_store is not None:
      self.n_store = n_store
      self.hp_hash = np.empty(self.n_store)
      self.choFactor = [[] for q in range(self.n_store)]
      self.logdetK = [[] for q in range(self.n_store)]
    if ep is not None: self.ep = np.array(ep)
    if fp is not None: self.fp = np.array(fp)
    if yerr is not None: self.yerr = np.array(yerr)

    #set fixed parameteres if ep set but not fp
    if self.fp is None and self.ep is not None:
      self.fp = ~(np.array(self.ep) > 0) * 1

    #try and set the number of mf/hyperparameters via mf/kf attributes/func if not defined directly
    self.n_par = self.p.size
    if self._n_mfp is None or mf is not None:
      try: self._n_mfp = self.mf.n_par(self.d)
      except:
        try: self._n_mfp = self.mf.n_par
        except: pass
        else: pass
      #else: print "#mf par set from mf.n_par(D)!"
    if self._n_hp is None or kf is not None:
      try: self._n_hp = self.kf.n_par(self.d)
      except:
        try: self._n_hp = self.kf.n_par
        except: pass
        #else: print "#kf par set from kf.n_par!"
      else: pass
    
    #set number of parameters for kernel and mean function
    #kernel function attributes will overwrite mf attributes
    if self._n_hp is not None:
      self.set_n_hp(self._n_hp)
    elif self._n_mfp is not None:
      self.set_n_mfp(self._n_mfp)
    # else: #if both are none
    #   self._n_mfp = 0
    #   self._n_hp = self.n_par - self._n_mfp

    #print warning if mf is set without knowing the number of parameters
    if mf is not None and self._n_mfp is None:
      print "warning: mean function was changed but n_hp or n_mfp is not set!"
      print "set one of them via set_pars, or eg set"

    #set kf function args and predictive args
    if self.x_pred is None: self.x_pred = self.x #set x_pred to x if pred not given
    #set mf function args and predictive args
    if self.xmf is None: #set mean function args
      self.xmf = self.kfVec() #set as first vector of x by default
      self.xmf_pred = self.kfVec_pred() #set as first vector of x by default
    else:
      if self.xmf_pred is None: self.xmf_pred = self.xmf

    #if x_pred is provided to set_pars, always reset xmf_pars if not also provided
    if x_pred is not None and xmf_pred is None:
      self.xmf_pred = self.kfVec_pred()

  def pars(self,p=None):
    """
    Simple function to return or set pars. Required as _pars is semi-private, and does not
    compute cho factor if set directly for cho kernel, eg MyGP._pars = [blah], plus should be a np.array.
    
    """

    if p is None:
      return np.copy(self._pars)
    else:
      # reset the hash thingy is pars are reset
      self._pars = np.array(p)
      self.n_par = self._pars.size
      self.hp_hash[self.si] = hash('') #create empty hash that won't be matched with any pars
      self.choFactor[self.si] = None
      self.logdetK[self.si] = None

  #set p as a property, ie can set p directly, eg MyGP.p = [blah...], and Pars will be called
  p = property(pars,pars)

  #add n_hp and n_mfp as properties, as they should be calculated from each other...
  def set_n_hp(self,n_hp=None):
    """
    Simple method to set or get n_hp, and sets n_mfp from it.
    """

    if n_hp is None:
      return self._n_hp
    else:
      self._n_hp = n_hp
      self._n_mfp = self.n_par - self._n_hp

  def set_n_mfp(self,n_mfp=None):
    """
    Simple method to set or get n_mfp, and sets n_hp from it.
    """

    if n_mfp is None:
      return self._n_mfp
    else:
      self._n_mfp = n_mfp
      self._n_hp = self.n_par - self._n_mfp

  #set n_hp and n_mfp as properties
  n_hp = property(set_n_hp,set_n_hp)
  n_mfp = property(set_n_mfp,set_n_mfp)

  def mfPars(self):
    "return the mean function parameters"
    return self._pars[:self.n_mfp]

  def kfVec(self,i=0):
    "Return the ith vector from the GP input matrix."
    return self.x.getA()[:,i]

  def kfVec_pred(self,i=0):
    "Return the ith vector from the GP predictive matrix."
    return self.x_pred.getA()[:,i]

  def describe(self):
    "Print the attributes of the GP object."

    print "--------------------------------------------------------------------------------"
    print "GP attributes:"
    print " Target values y:", self.y.shape #, type(self.y)
    print " GP input args x:", self.x.shape #, type(self.x)
    print " Log Prior:", self.logPrior.__name__
    print " Kernel Function:", self.kf.__name__
    print " Kernel Type:", self.kernel_type
    print " Hyperparameters:", self._pars[self._n_mfp:], "(#hp = {})".format(self.n_hp)
    if self.fp is not None: print " Fixed hyperparameters:", self.fp[self._n_mfp:]
    print " Mean Function:", self.mf.__name__
    print " MF args:", np.array(self.xmf).shape #, type(self.xmf)
    print " MF Parameters:", self._pars[:self.n_mfp], "(#mfp = {})".format(self.n_mfp)
    if self.fp is not None: print " Fixed MF Parameters:", self.fp[:self.n_mfp]
    print " Predictive GP args X:", np.array(self.x_pred).shape #, type(self.x_pred)
    print " Predictive mf args:", np.array(self.xmf_pred).shape #, type(self.xmf_pred)
    if self.yerr is not None: print " Target values err yerr:", np.array(self.yerr).shape #, type(self.yerr)
    print "--------------------------------------------------------------------------------"

  def logLikelihood_cholesky(self,p):
    "Function to calculate the log likeihood"

    #calculate the residuals
    r = self.y - self.mf(p[:self._n_mfp],self.xmf)

    #ensure r is an (n x 1) column vector
    r = np.matrix(np.array(r).flatten()).T

    #check if covariance, chol factor and log det are already calculated and stored
    new_hash = hash(p[-self.n_hp:].tostring()) # calculate and check the hash
    if np.any(self.hp_hash == new_hash):
      useK = np.where(self.hp_hash == new_hash)[0][0]
    else: #else calculate and store the new hash, cho_factor and logdetK
      useK = self.si = (self.si+1) % self.n_store #increment the store index number
#      self.choFactor[self.si] = LA.cho_factor(GPC.CovarianceMatrix(p[self._n_mfp:],self.x,KernelFunction=self.kf))
      self.choFactor[self.si] = LA.cho_factor(self.CovMat_p(p))
      self.logdetK[self.si] = (2*np.log(np.diag(self.choFactor[self.si][0])).sum())
      self.hp_hash[self.si] = new_hash
    
    #calculate the log likelihood
    logP = -0.5 * r.T * np.mat(LA.cho_solve(self.choFactor[useK],r)) - 0.5 * self.logdetK[useK] - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)

  def logLikelihood_toeplitz(self,p):
    """
    Function to calculate the log likeihood using a Toeplitz matrix
    Need to define a separate Toeplitz kernel
    """

    #evaluate mean function
    m = self.mf(p[:self._n_mfp],self.xmf)

    #calculate the residuals
    r = self.y - m

    #modify likelihood for multiplicative GP?
    logdetA2 = 0.
    if self.gp_type == 'mult':
      # Affine transform of GP
      # N(m,K) -> N(m,A*C*A), set K = A*C*A
      # need det(K) = det(A) * det(C) * det(A)
      # need to solve r^T K^-1 r = r^T (ACA)^-1 r = r^T A^-1 C^-1 A^-1 r,
      #   where C is Toeplitz, and ACA is not
      # therefore set r = r/m = A^-1 r and solve for C^-1 r as usual and get det(C)
      # then modify logdetC to logdetK

      r = r / m # modify r so r = A^-1 r (from affine transform of GP)

      logdetA2 = 2*np.log(m).sum()

    #ensure r is an (n x 1) column vector
    r = np.matrix(np.array(r).flatten()).T

    #get log determinant of covariance matrix and solve x = K^-1 r
    logdetK,self.teop_sol = GPT.LTZSolve(self.CovMat_p(p),r,self.teop_sol)

    #calculate the log likelihood
    logP = -0.5 * r.T * self.teop_sol - 0.5 * (logdetK + logdetA2) - (r.size/2.) * np.log(2*np.pi)
    
    return np.float(logP)

  def logLikelihood_white(self,p):
    "Function to calculate the log likeihood"

    #calculate the residuals
    r = self.y - self.mf(p[:self.n_mfp],self.xmf)

    #get the diagonal of the covariance matrix from the white noise kernel
    K = self.kf(self.yerr,p[self._n_mfp:])

    #calcaulte the log likelihood
    logP = - 0.5 * ( r**2 / K ).sum() - 0.5 * np.log(K).sum() - self.n/2.*np.log(2*np.pi)

    return logP

  @staticmethod #ensure static, so it can redefined using a 'normal' function
  def logPrior(p,nhp):
    """
    default log prior, keep hyperparameters > 0
    should be of the form logp(p,n_hp) where n_hp is number of hyperparameters
    egs:
    >> instance.logPrior = lambda *args: 0.
    >> instance.logPrior = lambda p,nhp: -np.inf if (np.array(p[-nhp:])<0).any() else 0.
    >> from scipy.stats.distributions import gamma,norm as norm_dist
    >> instance.logPrior = lambda p,nhp: np.log(norm_dist.pdf(p[6],.10,0.02)).sum()
    or combine them in a regular function...
    """

    #keep all kernel hyperparameters >=0
    return -np.inf if (np.array(p[-nhp:])<0).any() else 0.

  #define (log) posterior simply as the sum of likelihood and prior
  def logPosterior(self,p):
    "Function to calculate the log posterior"

    log_Prior = self.logPrior(p,self._n_hp)
    if log_Prior == -np.inf: return -np.inf #don't need to calculate logLikelihood if logPrior is -np.inf
    else: return self.logLikelihood(p) + log_Prior

  #default mean function - static method so can be redefined, just returns 0
  #add_n_par just adds a function attribute to specify number of mf parameters
  @staticmethod
  @GPU.add_n_par(0)
  def mf(*args):
    "default mean function = 0."
    return 0.

  def mfEval(self):
    "Returns the mean function evaluated at the current parameters"
    return self.mf(self._pars[:self._n_mfp], self.xmf)

  def mfEvalPred(self):
    "Returns the mean function evaluated at the current parameters"
    return self.mf(self._pars[:self._n_mfp], self.xmf_pred)

  def mfRes(self):
    "Returns the residuals from the mean function"
    return self.y - self.mf(self._pars[:self._n_mfp], self.xmf)

  def GPRes(self):
    "Return residuals from the GP + mf"
        
    #Construct the covariance matrix
    K = self.CovMat()
    K_s = self.CovMatBlock()
    K_ss = self.CovMatCornerDiag()
#    if self.kernel_type == 'Full':
#       K = GPC.CovarianceMatrix(self._pars[self._n_mfp:],self.x,KernelFunction=self.kf)
#       K_s = GPC.CovarianceMatrixBlock(self._pars[self._n_mfp:],self.x,self.x,KernelFunction=self.kf)
#       K_ss = GPC.CovarianceMatrixCornerDiag(self._pars[self._n_mfp:],self.x,KernelFunction=self.kf)
#     elif self.kernel_type == 'Toeplitz' or self.kernel_type == 'T':
#       K = GPT.CovarianceMatrixFullToeplitz(self._pars[self._n_mfp:],self.x,self.kf)
#       K_s = GPT.CovarianceMatrixBlockToeplitz(self._pars[self._n_mfp:],self.x,self.x,self.kf)
#       K_ss = GPT.CovarianceMatrixCornerDiagToeplitz(self._pars[self._n_mfp:],self.x,self.kf)

    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )

    #need do the regression on the *residual data* if mean funciton exists
    r = self.y - self.mf(self._pars[:self._n_mfp], self.xmf) #subtract the mean function
    
    #and do the regression on the residuals...
    y_pred, y_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)

    return self.y - self.mf(self._pars[:self._n_mfp], self.xmf) - y_pred

  def predictGP(self,x_pred=None,xmf_pred=None,wn=True):
    "Returns the predictive distributions for the GP alone using current hyperparmeters"

    if x_pred is not None: self.x_pred = x_pred
    if xmf_pred is not None: self.xmf_pred = xmf_pred

    if self.x is not self.x_pred and (self.kernel_type == 'Toeplitz' or self.kernel_type == 'T'):
      print "warning: using Toeplitz kernel for prediction only works when step sizes are equal" \
            " for x and x_pred.\nUse a 'Full' kernel after optimisation for if not."

    #Construct the covariance matrix
    K = self.CovMat()
    K_s = self.CovMatBlock()
    K_ss = self.CovMatCornerDiag(wn=wn)
#     if self.kernel_type == 'Full':
#       K = GPC.CovarianceMatrix(self._pars[self._n_mfp:],self.x,self.kf)
#       K_s = GPC.CovarianceMatrixBlock(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf)
#       K_ss = GPC.CovarianceMatrixCornerDiag(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)
#     elif self.kernel_type == 'Toeplitz' or self.kernel_type == 'T':
#       K = GPT.CovarianceMatrixFullToeplitz(self._pars[self._n_mfp:],self.x,self.kf)
#       K_s = GPT.CovarianceMatrixBlockToeplitz(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf)
#       K_ss = GPT.CovarianceMatrixCornerDiagToeplitz(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)

    #Calculate the precision matrix (needs optimised)
    PrecMatrix = np.linalg.inv( np.matrix(K) )

    #need do the regression on the *residual data* if mean funciton exists
    r = self.y - self.mf(self._pars[:self._n_mfp], self.xmf) #subtract the mean function

    #and do the regression on the residuals...
    y_pred, y_pred_err = GPR.GPRegress(K_s,PrecMatrix,K_ss,r)

    #return the predictive mean and variance
    return y_pred, y_pred_err

  def predict(self,x_pred=None,xmf_pred=None,wn=True,p=None):
    """
    Returns the predictive distributions for the GP and current hyperparmeters plus the
    mean function
    """

    #use provided parameters if given
    if p is not None:
      assert p.ndim == 1, "p should be 1D of length n_par"
      assert p.size == self.n_par, "p should be 1D of length n_par"
      p_save = self.pars()
      self.pars(p)

    #get the predictive distributions for the GP alone
    t_pred, t_pred_err = self.predictGP(x_pred=x_pred,xmf_pred=xmf_pred,wn=wn)

    #...and add the predictive mean function back
    t_pred += self.mf(self._pars[:self._n_mfp], self.xmf_pred)

    #and reset p to old values - bit inefficient but need to recode cov calculations otherwise
    if p is not None:
      self.pars(p_save)
    
    #return the predictive mean and variance
    return t_pred, t_pred_err

  def predictSample(self,p,x_pred=None,xmf_pred=None,wn=True,return_all=False):
    """
    Return predictive distributions for a collection of samples, p
    
    """
    
    #check dimensions of p
    assert p.ndim == 2, "p should be 2 dimensional"
    assert p.shape[1] == self.n_par, "p should be 2 dimensional, with 2nd dimension n_par"
    
    #create storage arrays
    N = p.shape[0]
    V,Verr = np.zeros((N,self.xmf_pred.size)),np.zeros((N,self.xmf_pred.size))
    
    #get random vectors
    for i in range(N):
      V[i],Verr[i] = self.predict(p=p[i],wn=wn,x_pred=x_pred,xmf_pred=xmf_pred)
    
    #get mean and standard deviation of the Gaussian mixture model
    mean = V.mean(axis=0)
    st_dev = np.sqrt(((V-mean)**2 + Verr**2).mean(axis=0))
    
    if return_all:
      return V,Verr
    else:
      #return the predictive mean and variance
      return mean,st_dev
      
  def optimise(self,method='NM',fp=None,**kwargs):
    """
    Optimise the parameters of the model - simple wrapper to Infer.Optimise
    """

    #print "Guess pars:", self._pars
    if fp is not None: self.fp = fp
    pars = OP.Optimise(self.logPosterior,self._pars,(),fixed=self.fp,method='NM',**kwargs)
    self.pars(pars)

  #create alias for optimise function
  opt = optimise

  def opt_global(self,ep=None,bounds=None,**kwargs):
    """
    Optimise the parameters of the model - simple wrapper to Infer.Optimise
    """
    
    #print "Guess pars:", self._pars
    if ep is not None: self.ep = ep
    
    if bounds is not None:
      pars = DE.DifferentialEvol(self.logPosterior,self._pars,(),bounds=bounds,**kwargs)
    else:
      if self.ep is not None:
        pars = DE.DifferentialEvol(self.logPosterior,self._pars,(),epar=self.ep,**kwargs)
      else:
        raise ValueError("ep is not defined in the GP, therefore ep or bounds must be provided")

    self.pars(pars)

  #create alias for optimise function
  #dev = opt_global
  #glob = opt_global

  def getRandomVector(self,p=None,wn=False):
    "Returns a random vector from the conditioned GP"

    #use provided parameters if given
    if p is not None:
      assert p.ndim == 1, "p should be 1D of length n_par"
      assert p.size == self.n_par, "p should be 1D of length n_par"
      p_save = self.pars()
      self.pars(p)

    # #set predictive distributions to the inputs if not set
    # if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
    # if self.mf_args_pred is None: self.mf_args_pred = self.mf_args

    #Construct the covariance matrix
    K = self.CovMat()
    K_s = self.CovMatBlock()
    K_ss = self.CovMatCorner(wn=wn)
#     if self.kernel_type == 'Full':
#       K = GPC.CovarianceMatrix(self._pars[self._n_mfp:],self.x,KernelFunction=self.kf)
#       K_s = GPC.CovarianceMatrixBlock(self._pars[self._n_mfp:],self.x_pred,self.x,KernelFunction=self.kf)
#       K_ss = GPC.CovarianceMatrixCornerFull(self._pars[self._n_mfp:],self.x_pred,KernelFunction=self.kf,WhiteNoise=wn)
#     elif self.kernel_type == 'Toeplitz' or self.kernel_type == 'T':
#       K = GPT.CovarianceMatrixFullToeplitz(self._pars[self._n_mfp:],self.x,self.kf)
#       K_s = GPT.CovarianceMatrixBlockToeplitz(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf)
#       K_ss = GPT.CovarianceMatrixCornerFullToeplitz(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)

    #get precision matrix
    PrecMatrix = np.linalg.inv( np.matrix(K) )

    #need do the regression on the *residual data* if mean funciton exists
    r = self.y - self.mf(self._pars[:self._n_mfp],self.xmf) #subtract the mean function

    #predictive mean
    m = self.mf(self._pars[:self._n_mfp], self.xmf_pred)

    #and reset p to old values - bit inefficient but need to recode cov calculations otherwise
    if p is not None:
      self.pars(p_save)
    
    return GPU.RandVectorFromConditionedGP(K_s,PrecMatrix,K_ss,r,m=m)
  
  def getRandomVectors(self,p,wn=False):
    """
    Returns a random vectors from the conditioned GP, using a range of parameter arrays
    
    p - 2d, N x K array, where N is the number of samples, and K is the number of parameters
      of the GP
    wn - include white noise?
    
    """
    
    #check dimensions of p
    assert p.ndim == 2, "p should be 2 dimensional"
    assert p.shape[1] == self.n_par, "p should be 2 dimensional, with 2nd dimension n_par"
    
    #create storage array
    N = p.shape[0]
    V = np.zeros((N,self.xmf_pred.size))
    
    #get random vectors
    for i in range(N):
      V[i] = self.getRandomVector(p=p[i],wn=wn)
          
    return V
  
  def predictDraws(self,p,wn=True):
    """
    Returns a random vectors from the conditioned GP, using a range of parameter arrays
    
    p - 2d, N x K array, where N is the number of samples, and K is the number of parameters
      of the GP
    wn - include white noise?
    
    """
        
    #get a random draw for each sample
    V = self.getRandomVectors(p=p,wn=wn)
    
    #and return the distribution
    return V.mean(axis=0),V.std(axis=0)
    
  def getRandomVectorFromPrior(self,wn=False):
    "Returns a random vector from the GP prior"

    # #set predictive distributions to the inputs if not set
    # if self.kf_args_pred is None: self.kf_args_pred = self.kf_args
    # if self.mf_args_pred is None: self.mf_args_pred = self.mf_args

    #Construct the covariance matrix
    K_ss = self.CovMatCorner(wn=wn)
#     if self.kernel_type == 'Full':
#       K_ss = GPC.CovarianceMatrixCornerFull(self._pars[self._n_mfp:],self.x_pred,KernelFunction=self.kf,WhiteNoise=wn)
#     elif self.kernel_type == 'Toeplitz' or self.kernel_type == 'T':
#       K_ss = GPT.CovarianceMatrixCornerFullToeplitz(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)

    return GPU.RandomVector(K_ss) + self.mf(self._pars[:self._n_mfp], self.xmf_pred)

  #############################################################################################################
  #use dill to save current state of gp
  def save(self,filename):
    """Save the current state of the GP to a file using dill"""
    if not dill_available:
      print "dill module not available. can't save gp"
    else:
      file = open(filename,'w')
      dill.dump(self,file)
      file.close()
  
  #############################################################################################################
  #quicklook plotting functions
  
  def plotRanges(self,wn=True,**kwargs):
    """Plots the 1 and 2 sigma range of the GP (but doesn't take into account mean function errors)"""

    t_pred,t_pred_err = self.predict(wn=wn)
    GPU.PlotRanges(self.xmf_pred,t_pred,t_pred_err,**kwargs)
  
  def plotData(self,**kwargs):
    """Plots the data with errorbars"""
  
    #set the errors
    err = self._pars[-1]
    #else: err = self._pars[self.n_hp-1]
  
    GPU.PlotData(self.xmf,self.y,np.ones(self.y.size)*err,title=None,**kwargs)
  
  def plotMean(self,ax=None):
    """Plots the mean function (with predictive arguments)"""
    
    #plot the mean function
    if ax==None: ax = pylab.gca()
    ax.plot(self.xmf_pred,self.mfEvalPred()*np.ones(self.xmf_pred.size),'r--')
  
  def plot(self,wn=True,ax=None,**kwargs):
    """Convenience method to call both plotRanges, plotData and plotMean with defaults"""
  
    if self.kernel_type is not 'White' and self.kernel_type is not 'W':
      self.plotRanges(wn=wn,ax=ax)
    self.plotData(ax=ax)
    self.plotMean(ax=ax)
  
  #############################################################################################################

  #wrappers for normal covariance matrix functions
  def CovarianceMatrixAdd_p(self,p):
    """return covariance matrix for normal kernel given full parameter set p"""
    
    K = GPC.CovarianceMatrix(p[self._n_mfp:],self.x,KernelFunction=self.kf)
    return K

  def CovarianceMatrixFullAdd(self):
    """return covariance matrix for normal kernel using current stored parameters"""

    K = GPC.CovarianceMatrix(self._pars[self._n_mfp:],self.x,KernelFunction=self.kf)
    return K

  def CovarianceMatrixBlockAdd(self):
    """return covariance matrix block for normal kernel, ie training points vs predictive points"""

    K_s = GPC.CovarianceMatrixBlock(self._pars[self._n_mfp:],self.x_pred,self.x,KernelFunction=self.kf)
    return K_s

  def CovarianceMatrixCornerAdd(self,wn=True):
    """return covariance matrix corner for normal kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPC.CovarianceMatrixCornerFull(self._pars[self._n_mfp:],self.x_pred,KernelFunction=self.kf,WhiteNoise=wn)
    return K_ss
  
  def CovarianceMatrixCornerDiagAdd(self,wn=True):
    """return diagonal of covariance matrix corner for normal kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPC.CovarianceMatrixCornerDiag(self._pars[self._n_mfp:],self.x_pred,KernelFunction=self.kf,WhiteNoise=wn)
    return K_ss
  
  #############################################################################################################
  #wrappers for toeplitz covariance matrix functions
  def CovarianceMatrixToeplitzAdd_p(self,p):
    """return covariance matrix for toeplitz kernel given full parameter set p"""

    K = GPT.CovarianceMatrixToeplitz(p[self._n_mfp:],self.x,self.kf)
    return K

  def CovarianceMatrixFullToeplitzAdd(self):
    """return covariance matrix for toeplitz kernel using current stored parameters"""

    K = GPT.CovarianceMatrixFullToeplitz(self._pars[self._n_mfp:],self.x,self.kf)
    return K
 
  def CovarianceMatrixBlockToeplitzAdd(self):
    """return covariance matrix block for toeplitz kernel, ie training points vs predictive points"""

    K_s = GPT.CovarianceMatrixBlockToeplitz(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf)
    return K_s

  def CovarianceMatrixCornerToeplitzAdd(self,wn=True):
    """return covariance matrix corner for toeplitz kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPT.CovarianceMatrixCornerFullToeplitz(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)
    return K_ss

  def CovarianceMatrixCornerDiagToeplitzAdd(self,wn=True):
    """return diagonal of covariance matrix corner for toeplitz kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPT.CovarianceMatrixCornerDiagToeplitz(self._pars[self._n_mfp:],self.x_pred,self.kf,WhiteNoise=wn)
    return K_ss

  #############################################################################################################
  #wrappers for multiplicative/normal covariance matrix functions
  def CovarianceMatrixMult_p(self,p):
    """return covariance matrix for normal kernel given full parameter set p"""
    
    K = GPMC.CovarianceMatrixMult(p[self._n_mfp:],self.x,self.kf,
      self.mf,p[:self._n_mfp],self.xmf)
    return K

  def CovarianceMatrixFullMult(self):
    """return covariance matrix for normal kernel using current stored parameters"""

    K = GPMC.CovarianceMatrixMult(self._pars[self._n_mfp:],self.x,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf)
    return K

  def CovarianceMatrixBlockMult(self):
    """return covariance matrix block for normal kernel, ie training points vs predictive points"""

    K_s = GPMC.CovarianceMatrixBlockMult(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,self.xmf)
    return K_s

  def CovarianceMatrixCornerMult(self,wn=True):
    """return covariance matrix corner for normal kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPMC.CovarianceMatrixCornerFullMult(self._pars[self._n_mfp:],self.x_pred,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,WhiteNoise=wn)
    return K_ss
  
  def CovarianceMatrixCornerDiagMult(self,wn=True):
    """return diagonal of covariance matrix corner for normal kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPMC.CovarianceMatrixCornerDiagMult(self._pars[self._n_mfp:],self.x_pred,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,WhiteNoise=wn)
    return K_ss
  
###############################################################################################################
  #wrappers for multiplicative/toeplitz covariance matrix functions
#   def CovarianceMatrixToeplitzMult_p(self,p):
#     """return covariance matrix for toeplitz kernel given full parameter set p"""
#     
#     K = GPT.CovarianceMatrixToeplitz(p[self._n_mfp:],self.x,self.kf)
#     return K

  def CovarianceMatrixFullToeplitzMult(self):
    """return covariance matrix for toeplitz kernel using current stored parameters"""

    K = GPT.CovarianceMatrixFullToeplitzMult(self._pars[self._n_mfp:],self.x,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf)

    return K

  def CovarianceMatrixBlockToeplitzMult(self):
    """return covariance matrix block for toeplitz kernel, ie training points vs predictive points"""

    K_s = GPT.CovarianceMatrixBlockToeplitzMult(self._pars[self._n_mfp:],self.x_pred,self.x,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,self.xmf)
    return K_s

  def CovarianceMatrixCornerToeplitzMult(self,wn=True):
    """return covariance matrix corner for toeplitz kernel, ie predictive points cov with themselves, white noise optional"""
    
    K_ss = GPT.CovarianceMatrixCornerFullToeplitzMult(self._pars[self._n_mfp:],self.x_pred,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,WhiteNoise=wn)
    return K_ss

  def CovarianceMatrixCornerDiagToeplitzMult(self,wn=True):
    """return diagonal of covariance matrix corner for toeplitz kernel, ie predictive points cov with themselves, white noise optional"""

    K_ss = GPT.CovarianceMatrixCornerDiagToeplitzMult(self._pars[self._n_mfp:],self.x_pred,self.kf,
      self.mf,self._pars[:self._n_mfp],self.xmf_pred,WhiteNoise=wn)
    return K_ss

  #############################################################################################################
