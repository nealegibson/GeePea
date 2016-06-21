
import numpy as np
import time
from scipy.optimize import fmin,brute,fmin_cg,fmin_powell,fmin_bfgs,fmin_l_bfgs_b

def Optimise(LogLikelihood,par,func_args,fixed=None,type='max',method='NM',maxiter=10000, maxfun=10000, verbose=True, bounds=None):
  """
  Function wrapper to find the maximum (or min) of a function using the scipy fmin-like
  minimisation routines, allowing some of the parameters to be fixed. This requires a
  messy wrapper because none of the parameter vector passed to the fmin funcitons can be
  fixed!
  
  LogLikelihood - function to optimise, of the form func(parameters, func_args). Doesn't
    need to be a log likelihood of course - just that's what I use it for!
  par - array of parameters to the func to optimise
  func_args - additional arguments to the func - usually as a tuple
  type - either max or min to optimise the funciton
  method - algorithm to use NM - Nelder-Mead (Downhill simplex/Amoeba), CG - Conjugate-gradient,
    or P - Powell's method
    ***CG and P not working properly for some reason!!***
  maxiter, maxfun - max iterations and function evaluations for the nelder-mead algorithm
  """
  
  if fixed is None:
    var_par = np.copy(par)
    fixed_par = np.zeros(var_par.size)
  #otherwise construct the parameter vector from var_par and fixed_par_val
  else:
    par = np.array(par)
    fixed = np.array(fixed) #ensure fixed is a np array
    #assign parameters to normal param vector
    fixed_par = par[np.where(fixed==True)]
    var_par = par[np.where(fixed!=True)]
  
  #set the algorithm to use - CG and P not working (at least not well)
  add_kwords = {'verbose':verbose}
  if method == 'NM':
    Algorithm = NelderMead
    add_kwords = {'maxiter':maxiter, 'maxfun':maxfun,'verbose':verbose}
  elif method == 'CG':
    print "warning: CG method didn't work properly during testing"
    Algorithm = ConjugateGradient
  elif method == 'P':
    print "warning: Powell algorithm didn't work properly during testing"
    Algorithm = Powell
  elif method == 'BFGS':
#    print "warning: Powell algorithm didn't work properly during testing"
    Algorithm = BFGS
  elif method == 'L-BFGS-B':
#    print "warning: Powell algorithm didn't work properly during testing"
    Algorithm = L_BFGS_B
    add_kwords = {'verbose':verbose,'bounds':bounds}
  else:
    print "error: optimisation function not found"
    return par
  
  #set the optimisation function to pos or neg for the fmin funcitons
  if type == 'max': OptFunc = NegFixedPar_func
  elif type == 'min': OptFunc = FixedPar_func
  else:
    print "error: %s not a valid option" % type
    return par
  
  #call the optimser with the appropriate function
  fitted_par = Algorithm(OptFunc, var_par, (LogLikelihood,func_args,fixed,fixed_par), \
      **add_kwords)
  
  #now return the params in the correct order...
  if fixed is None or fixed.sum()==0:
    return_par = fitted_par
  else:
    return_par = np.copy(par)
    return_par[np.where(fixed!=True)] = fitted_par
    
  return return_par

##########################################################################################
#Fuctions to optimise - these need the var_par arg to all be variable pars, hence the horrible
#wrappers needed!
def NegFixedPar_func(var_par,func,func_args,fixed=None,fixed_par=None,**kwargs):
  
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
  
  #now call the function as normal and return the neg, expanding the args and kwargs
  return -func(par,*func_args,**kwargs)

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
# Optimisation functions with timers
def NelderMead(ErrFunc, params0, function_args, maxiter=10000, maxfun=10000, verbose=True):

  if verbose:
    print "Running Nelder-Mead simplex algorithm... "
    t0 = time.clock()
    disp = True
  else:
    disp = False
  params = fmin(ErrFunc, params0, args=function_args,maxiter=maxiter, maxfun=maxfun,disp=disp)
  if verbose:
    print "(Time: %f secs)" % (time.clock()-t0)
    print "Optimised parameters: ", params,"\n"
  return params

def Powell(ErrFunc, params0, function_args, verbose=True):

  if verbose:
    print "Running Powell's method minimisation... "
    t0 = time.clock()
  params = fmin_powell(ErrFunc, params0, args=function_args)
  if verbose:
    print "(Time: %f secs)" % (time.clock()-t0)
    print "Optimised parameters: ", params,"\n"
  return params

def ConjugateGradient(ErrFunc, params0, function_args, verbose=True):

  if verbose:
    print "Running conjugate gradient minimisation... "
    t0 = time.clock()
  params = fmin_cg(ErrFunc, params0, args=function_args)
  if verbose:
    print "(Time: %f secs)" % (time.clock()-t0)
    print "Optimised parameters: ", params,"\n"
    
  return params

def BFGS(ErrFunc, params0, function_args, verbose=True):

  if verbose:
    print "Running BFGS minimisation... "
    t0 = time.clock()
  params = fmin_bfgs(ErrFunc, params0, args=function_args,full_output=1,gtol=1e-5)
  if verbose:
    print "(Time: %f secs)" % (time.clock()-t0)
    print "Optimised parameters: ", params[0],"\n"
    
  return params[0]

def L_BFGS_B(ErrFunc, params0, function_args, verbose=True, **kw):
  """
  This can include a 'bound' keyword of tuple pairs (None,None) for each parameter, None
    indicates no upper/lower limit, or else add a limit. Haven't fully tested this
  """
  if verbose:
    print "Running L-BFGS-B minimisation... "
    t0 = time.clock()
  params,min_val,dict  = fmin_l_bfgs_b(ErrFunc, params0,approx_grad=1, args=function_args, bounds=kw['bounds'])
  print dict
  if verbose:
    print "(Time: %f secs)" % (time.clock()-t0)
    print "Optimised parameters: ", params,"\n"
    
  return params

##########################################################################################
