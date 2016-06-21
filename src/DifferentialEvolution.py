
import numpy as np
from scipy.optimize import differential_evolution

##########################################################################################

def DifferentialEvol(LogLikelihood,par,func_args,epar=None,bounds=None,type='max',Nsig=3,verbose=True,**kwargs):
  """
  Function wrapper to find the maximum (or min) of a function using the scipy differential
  evolution function. Wrapper required for easy interface, adn to enable fixed paramameters.
  This isn't strictly necessary as the bounds can be set to be infinitely thin, but much
  more function evaluations are taken otherwise.
  
  LogLikelihood - function to optimise, of the form func(parameters, func_args). Doesn't
    need to be a log likelihood of course - just that's what I use it for!
  par - array of parameters to the func to optimise
  func_args - additional arguments to the func - usually as a tuple
  epar - array of parameter errors
  bounds - list of tuples/lists containing parameter ranges eg [(1,2),0,,None,(4,5)].
    A 0 (must be integer) or 'None' instead of a tuple/list indicates a fixed parameter.
    'bounds' overwrites 'epar'.
  Nsig - no of sigma from errbars and par to set bounds
  type - either max or min to optimise the funciton
  
  """
  
  if epar is None and bounds is None:
    raise ValueError("error: epar or bounds must be defined!")
  
  #first define fixed parameter array
  if bounds is not None:
    fixed = np.array([0 if (i is not 0) and (i is not None) else 1 for i in bounds]) == 1
  else:
    fixed = np.array(epar) == 0
  
  if bounds is not None: #construct bounds from provided boundaries
    if fixed.sum() == 0:
      #fixed = None
      par = np.array(par)
      var_par = np.copy(par)
      fixed_par = np.zeros(var_par.size)
    #otherwise construct the parameter vector from var_par and fixed_par_val
    else:
      par = np.array(par)
      fixed = np.array(fixed) #ensure fixed is a np array
      #assign parameters to normal param vector
      fixed_par = par[np.where(fixed==True)]
      var_par = par[np.where(fixed!=True)]
    #set the bounds for the varied parameters
    bounds_var = [tup for tup in bounds if (tup is not 0) and (tup is not None)]
  else: #construct bounds from epar
    #constuct the var_par and fixed_par arrays
    if fixed.sum() == 0:
      #fixed = None
      par = np.array(par)
      var_par = np.copy(par)
      var_epar = np.copy(epar)
      fixed_par = np.zeros(var_par.size)
    #otherwise construct the parameter vector from var_par and fixed_par_val
    else:
      par = np.array(par)
      epar = np.array(epar)
      fixed = np.array(fixed) #ensure fixed is a np array
      #assign parameters to normal param vector
      fixed_par = par[np.where(fixed==True)]
      var_par = par[np.where(fixed!=True)]
      var_epar = epar[np.where(fixed!=True)]
    #set the bounds for the varied parameters
    bounds_var = [(p-Nsig*ep,p+Nsig*ep) for p,ep in zip(var_par,var_epar)]
  
  assert type == 'max' or type == 'min', "type must be max or min"
  if type == 'max': OptFunc = NegFixedPar_func
  elif type == 'min': OptFunc = FixedPar_func
  
  #redefine for fixed parameters
  if verbose:
    print "-"*80
    print "Differential Evolution parameter ranges:"
    for i in range(par.size):
      print " p[{}] => {}".format(i,'fixed' if fixed[i] else bounds_var[(~fixed[:i]).sum()])
  
  #run the DE algorithm, without finishing algorithm
  if fixed.sum() == None:
    fixed = None
    fixed_par = None
  DE = differential_evolution(OptFunc,bounds_var,args=(LogLikelihood,func_args,fixed,fixed_par),polish=False,**kwargs)
  fitted_par = DE.x

  #print out results
  if verbose:
    print "No of function evaluations = {}".format(DE.nfev)
    print "DE {} @ {}".format(type,DE.x)
    print "-"*80
  
  #now return the params in the correct order...
  if fixed is None or fixed.sum()==0:
    return_par = fitted_par
  else:
    return_par = np.copy(par)    
    return_par[np.where(fixed!=True)] = fitted_par

  #return the optimised position
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
#create function aliases

DiffEvol = DifferentialEvol
DE = DifferentialEvol

##########################################################################################

