
import numpy as np
import time
from scipy.optimize import fmin,brute,fmin_cg,fmin_powell
from scipy.optimize import leastsq

import MyFuncs as MF

from leastsqbound import leastsqbound

##########################################################################################

def LevMar(func,par,func_args,y,err=None,fixed=None,bounds=None,return_BIC=False,return_AIC=False):#,maxiter=10000, maxfun=10000, verbose=True):
  """
  Function wrapper for Levenberg-Marquardt via scipy.optimize.leastsq
  
  Similar interface to Optimiser.py to allow for fixed parameters. I have included
  code from https://github.com/jjhelmus/leastsqbound-scipy/ in order to implement the
  bound case (leastsqbound), else defaults to leastsq (which leastsqbound does anyway!)
  
  Has a similar syntax to Optimiser, but requires the function to be passed rather than
  the likelihood function (therefore cannot optimise noise parameters, GPs etc). Added
  leastsqbound (from https://github.com/jjhelmus/leastsqbound-scipy/, Copyright (c) 2012
  Jonathan J. Helmus, see file for full license) that uses parameter transformations to
  enable bound optimisation. This is only enabled for bound optimisation. The error
  estimates from LM optimisation are useful to seed MCMC, and also can be used to compute
  an evidence approximation (via BIC, AIC or Laplace optimisation). Care needs to be taken
  as to the reliability of the uncertainties compared to an MCMC, and this should
  generally only be used as an approximation.
  
  Parameters
  ----------
  func : function to fit
  par : parameters of function
  func_args : arguments to function
  y : measurements
  err : uncertainties for each measurement, defaults to array of ones
  fixed : array of fixed parameters
  bounds : array of boundaries (min,max) pairs for each parameter, array (N_param x 2)
    - input None where for no lower/upper boundary. bounds = None uses std leastsq
  return_BIC : return BIC evidence estimate
  return_AIC : return AIC evidence estimate
  
  Returns
  -------
  bf_par : fitted parameter vector
  err_par : uncertainty estimate for each parameter
  rescale : rescale value for errors, equivalent to white noise estimate for err = 1
      std of the residuals in this case, or if err provided a rescale to get true noise
  K_fit : covariance matrix of the fitted parameters (should edit to include non-fitted terms?)
      edits will be needed to use in laplace optimisation of alpha parameters...
  logE : evidence approximation from Laplace approximation
  logE_BIC : evidence approximation from BIC (optional)
  logE_AIC : evidence approximation from AIC (optional)
  
  """
  
  #make variable and fixed par arrays
  if fixed is None:
    var_par = np.copy(par)
    fixed_par = None
  #otherwise construct the parameter vector from var_par and fixed_par_val
  else:
    par = np.array(par)
    fixed = np.array(fixed) #ensure fixed is a np array
    #assign parameters to normal param vector
    fixed_par = par[np.where(fixed==True)]
    var_par = par[np.where(fixed!=True)]
  
  #set error vector if not provided
  if err is None: err = np.ones(y.size)
  else: err = err * np.ones(y.size)
  
  #get the bounds for variable parameters
  if bounds is None:
    bounds_var = None
  else:
    bounds_var = bounds[np.where(fixed!=True)]
  
  #perform the optimisation:  
  if bounds is None: R = leastsq(LM_ErrFunc,var_par,(func,func_args,y,err,fixed,fixed_par),full_output=1)
  else: R = leastsqbound(LM_ErrFunc,var_par,(func,func_args,y,err,fixed,fixed_par),bounds=bounds_var,full_output=1)
  
  fitted_par = R[0]
  K_fit = R[1]
  
  #reconstruct the full parameter vector and covariance matrix
  if fixed is None:
    bf_par = fitted_par
    return_err = np.sqrt(np.diag(K_fit))
    K = K_fit
  else:
    bf_par = np.copy(par)    
    bf_par[np.where(fixed!=True)] = fitted_par
    err_par = np.zeros(par.size)
    err_par[np.where(fixed!=True)] = np.sqrt(np.diag(K_fit))
    K = K_fit
  
  #rescale errors and covariance
  rescale = np.std((y - func(bf_par,*func_args)) / err)
  K_fit *= rescale**2
  err_par *= rescale
  
  #estimate the white noise from the residuals
  resid = y - func(bf_par,*func_args)
  wn = np.std(resid)
  
  print "LM fit parameter estimates:"
  print " par = mean +- err"
  for i in range(bf_par.size): print " p[{}] = {:.8f} +- {:.8f}".format(i,bf_par[i],err_par[i])
  print "white noise =", wn
  
  #calculate the log evidence for the best fit model
  logP_max = MF.LogLikelihood_iid(resid,1.,err*rescale)
  D = np.diag(K_fit).size
  N_obs = y.size
  logE_BIC = logP_max - D/2.*np.log(N_obs)
  logE_AIC = logP_max - D * N_obs / (N_obs-D-1.)
  sign,logdetK = np.linalg.slogdet( 2*np.pi*K_fit ) # get log determinant
  logE = logP_max + 0.5 * logdetK #get evidence approximation based on Gaussian assumption
  
  #expand K to the complete covariance matrix - ie even fixed parameters + white noise
  ind = np.hstack([(fixed==0).cumsum()[np.where(fixed==1)],K.diagonal().size]) #get index to insert zeros
  Kn = np.insert(np.insert(K,ind,0,axis=0),ind,0,axis=1) #insert zeros corresponding to fixed pars
  
  print "Gaussian Evidence approx:"
  print " log ML =", logP_max
  print " log E =", logE
  print " log E (BIC) =", logE_BIC, "(D = {}, N = {})".format(D,N_obs)
  print " log E (AIC) =", logE_AIC, "(D = {}, N = {})".format(D,N_obs)
  
  ret_list = [bf_par,err_par,rescale,Kn,logE]
  if return_BIC: ret_list.append(logE_BIC)
  if return_AIC: ret_list.append(logE_AIC)
  return ret_list


##########################################################################################
#create some aliases
LeastSQ = LevMar
LM = LevMar

##########################################################################################

def LM_ErrFunc(var_par,func,func_args,y,err,fixed=None,fixed_par=None):
  """
  Error function for LM optimisation. Uses FixedPar_func to wrap functions and allow
  fixed parameters
  
  """
  
  return (y - FixedPar_func(var_par,func,func_args,fixed,fixed_par)) / err

##########################################################################################
def FixedPar_func(var_par,func,func_args,fixed=None,fixed_par=None,**kwargs):
  
  #if no fixed parameters passed - just assign var_par to par and call function
  if fixed is None:
    par = np.copy(var_par)
  #otherwise construct the parameter vector from var_par and fixed_par_val
  else:
    fixed = np.array(fixed) #ensure fixed is a np array
    par = np.empty(fixed.size) #create empty pars array
    #assign parameters to normal param vector
    par[np.where(fixed==True)] = fixed_par
    par[np.where(fixed!=True)] = var_par
  
  #now call the function as normal, expanding the args and kwargs
  return func(par,*func_args,**kwargs)
##########################################################################################


