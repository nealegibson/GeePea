
from __future__ import print_function

from . import Optimiser as OP
from . import DifferentialEvolution as DE
import numpy as np

class combine(object):
  """
  Wrapper class to receive a list of gps, and combine to enable joint fits.
  
  The main function is to use the order parameter (list of independent parameters) and
  construct full lists of parameters/errors/etc that can be used directly within
  optimisation functions. order is a list indicating the parameter number for each
  gp. Where the same no is given, parameters are considered as common, otherwise
  independent. e.g. if order1 = [0,1,2,3] and order2 = [0,1,4,5] the first two parameters
  would be common (0,1), and the rest would be independent. The full parameter array
  would contain 6 parameters. order can be given directly to the initaliser, or each
  gp can be assigned an order parameter (ie gp.order = [x,y,z...]). Some funcitons
  assume that fp (fixed pars) are also available from each GP.
  
  Note that the individial GPs will each use their prior distributions, so need to be
  careful these are not used twice. An overall prior can also be added to this function
  via logPrior. The method logLikelihoodPrior will ignore individual priors for each GP,
  but still add the overall logPrior (default = 0).
  
  """
  
  def __init__(self,gps,order=None):
    """
    Initialises the combined GP. Main function is to use 'order' to set/get full
    parameter lists including errors, fixed pars etc...
    """
    self.gps = gps #list of gps
    if order is None:
      try:
        self.order = [gp.order for gp in gps] #try to set from gp attributes
      except:
        raise Exception, "order not found within gps"
    else:
      self.order = order
    self.n = np.concatenate(self.order).max()+1 #get number of parameters
    self._pars = np.empty(self.n) #set parameter array
    self._epars = np.empty(self.n) #set parameter array
    
    #set parameters from gps directly
    self.set_pars()
    self.set_epars()
    
  def set_pars(self):
    "Sets the parameters from the individual GPs"
    for gp,ord in zip(self.gps,self.order):
      self._pars[ord] = gp.p #set parameters
  
  def set_epars(self):
    "Sets the error parameters from the individual GPs"
    for gp,ord in zip(self.gps,self.order):
      self._epars[ord] = gp.ep #set parameters
  
  def pars(self,p=None):
    "Gets/Sets the parameters from full list of all parameters"
    if p is None:
      return np.copy(self._pars)
    else:
      self._pars = np.array(p) #set class parameters
      #and also set for each gp
      for gp,ord in zip(self.gps,self.order):
        gp.p = self._pars[ord] #set parameters
      
  def epars(self,ep=None):
    "Gets/Sets the error parameters from full list of all parameters"
    if ep is None:
      return np.copy(self._epars)
    else:
      self._epars = np.array(ep) #set class parameters
      #and also set for each gp
      for gp,ord in zip(self.gps,self.order):
        gp.ep = self._epars[ord] #set parameters

  #set p as a property, ie can set p directly, eg MyGPs.p = [blah...], and Pars will be called
  p = property(pars,pars)
  ep = property(epars,epars)
  
  @staticmethod #ensure static, so it can redefined using a 'normal' function
  def logPrior(p):
    """
    logPrior called by logPosterior. Defined here as a static func so can be easily
    redefined outside the object.
    should be of the form logp(p) - diff from with gp class as don't know the no of
    hyperparameters. Usually, the priors in each individual GP can be used, but need to
    be careful not to 'double count' any for common parameters
    egs:
    >> instance.logPrior = lambda *args: 0.
    >> instance.logPrior = lambda p: -np.inf if (np.array(p[-nhp:])<0).any() else 0.
    >> from scipy.stats.distributions import gamma,norm as norm_dist
    >> instance.logPrior = lambda p: np.log(norm_dist.pdf(p[6],.10,0.02)).sum()
    or combine them in a regular function...
    """

    #return 0 unless otherwise defined
    return 0.
  
  def logPosterior(self,p):
    "Computes the log posterior for each gp, and sums."
    
    #call logPosterior for each GP using input pars
    logP = np.array([gp.logPosterior(p[ord]) for gp,ord in zip(self.gps,self.order)]).sum()
    
    return logP + self.logPrior(self._pars)

  def logLikelihoodPrior(self,p):
    "Computes the log posterior for each gp, and sums."
    
    #call logPosterior for each GP using input pars
    logP = np.array([gp.logLikelihood(p[ord]) for gp,ord in zip(self.gps,self.order)]).sum()
    
    return logP + self.logPrior(self._pars)
  
  def optimise(self,method='NM',**kwargs):
    "Constructs the fixed parameter array from the individual gps, and calls an optimiser."
    self.fp = np.zeros(self.p.size)
    
    #set fixed parameter array from gps
    try:
      for gp,ord in zip(self.gps,self.order):
        self.fp[ord] = gp.fp #set parameters
    except:
      raise Exception, "Cannot set fixed parameters from gps."
    
    #optimise the logPosterior
    pars = OP.Optimise(self.logPosterior,self._pars,(),fixed=self.fp,method=method,**kwargs)
    
    #set parameters for all gps
    self.pars(pars)
      
  #create alias for optimise function
  opt = optimise
  
  def opt_global(self,ep=None,bounds=None,**kwargs):
    "Constructs the bounds parameter array from the individual gps, and calls an optimiser."
    print("Function not yet implemented!")
    
    if bounds is None:
      print("trying to set bounds!")
      #try to set from individual gp bounds
      try:
        all_bounds = np.concatenate([gp.bounds for gp in self.gps])
        order = np.concatenate([gp.order for gp in self.gps])
        self.bounds = [[],]*(self.p.size)
        for i in range(self.p.size):
          self.bounds[i] = all_bounds[np.where(order==i)[0][0]]
      except:
        raise Exception, "Cannot set bounds from gps."      
    else: #set from bound if provided
      self.bounds = bounds
    
    #optimise the logPosterior    
    pars = DE.DifferentialEvol(self.logPosterior,self._pars,(),bounds=self.bounds,**kwargs)
  
    #set parameters for all gps
    self.pars(pars)
