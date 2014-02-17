
import numpy as np
import time
import pylab
from scipy.optimize import fmin,brute,fmin_cg,fmin_powell
from scipy.optimize import fsolve

from Optimiser import *

def FixedPar_func_offset(var_par,max_loglik,*arglist):
  """
  Simple function to enable root finding for where logL_max - 0.5
  
  """
  
  return FixedPar_func(var_par,*arglist) - max_loglik + 0.5

def PlotConditionals(LogLikelihood,par,err,low,upp,func_args=(),plot_samp=100,opt=False,par_in=None,wait=False):
  """
  Plot the conditional distributions for each variable parameter. Used to visualise the
  conditional errors, and get sensible inputs to ConditionalErrors function.
  
  """
  
  #first optimise the log likelihood?
  if opt: op_par = Optimise(LogLikelihood,par[:],func_args,fixed=(np.array(err) == 0)*1)
  else: op_par = np.copy(par)
  
  max_loglik = LogLikelihood(op_par,*func_args)
  
  if par_in == None: par_in = np.where(np.array(err) != 0.)[0]
  
  for i in par_in:
   
   par_range = np.linspace(low[i],upp[i],plot_samp)
   log_lik = np.zeros(plot_samp)
   temp_par = np.copy(op_par)
   for q,par_val in enumerate(par_range):
     temp_par[i] = par_val
     log_lik[q] = LogLikelihood(temp_par,*func_args)
   pylab.clf()
   pylab.plot(par_range,log_lik)
   pylab.plot(par_range,max_loglik-(par_range-op_par[i])**2/2./err[i]**2,'r--')
   pylab.axvline(op_par[i],color='r')
   pylab.axvline(op_par[i]+err[i],color='g')
   pylab.axvline(op_par[i]-err[i],color='g')
   pylab.axhline(max_loglik-0.5,color='g',ls='--')
   pylab.xlabel("p[%s]" % str(i))
   pylab.ylabel("log Posterior")
   #pylab.xlims(low[i],upp[i])
   if wait: raw_input("")  

def PlotSlice(LogLikelihood,par,low,upp,par_in,func_args=(),plot_samp=100):
  """
  Plot the conditional distributions for each variable parameter. Used to visualise the
  conditional errors, and get sensible inputs to ConditionalErrors function.
  
  """
  
  i = par_in
  op_par = np.copy(par)
  max_loglik = LogLikelihood(op_par,*func_args)  
  
  par_range = np.linspace(low,upp,plot_samp)
  log_lik = np.zeros(plot_samp)
  temp_par = np.copy(op_par)
  for q,par_val in enumerate(par_range):
    temp_par[i] = par_val
    log_lik[q] = LogLikelihood(temp_par,*func_args)
  print np.exp(log_lik-max_loglik)
  pylab.clf()
  pylab.subplot(211)
  pylab.plot(par_range,log_lik)
  pylab.axhline(max_loglik-0.5,color='g',ls='--')
  pylab.xlabel("p[%s]" % str(i))
  pylab.ylabel("log Posterior")
  pylab.subplot(212)
  pylab.plot(par_range,np.exp(log_lik-max_loglik))
  pylab.axvline(op_par[i],color='r')
  pylab.axhline(0.6065,color='g',ls='--')
  pylab.xlabel("p[%s]" % str(i))
  pylab.ylabel("Posterior")

def ConditionalErrors(LogLikelihood,par,err,func_args=(),plot=False,plot_samp=100,opt=False):
  """
  Function to find the range of conditional distributions for each variable parameter, ie
  vary each parameter until delta chi_2 = 1.
  
  Cycles through each parameter in turn, optimises with respect to that parameter and then
  finds where the log likelihood changes by -0.5 in each direction. Returns the average
  of the plus/minus errors plus the new optimised values. Optionally optimise the function
  with respect to all variables to start.
  
  LogLikelihood - Log likelihood function
  par - parameter array
  err - error array, used to determine the variables and provides guesses for the root
    finder
  func_args - tuple of function arguments
  plot - make plots of each dimension
  plot_samp - no of samples for each plot
  opt - do global optimisation to start?
  
  """
  
  #first optimise the log likelihood?
  if opt: op_par = Optimise(LogLikelihood,par[:],func_args,fixed=(np.array(err) == 0)*1)
  else: op_par = np.copy(par)
  old_op_par = np.copy(op_par)
  err_pos = np.zeros(len(par))
  err_neg = np.zeros(len(par))
  
  #loop through the variables
  for i in np.where(np.array(err) != 0.)[0]:
    
    #create fixed and var par arrays
    fixed = (np.arange(len(err))!=i)*1
    fixed_par = op_par[np.where(fixed==True)]
    var_par = op_par[np.where(fixed!=True)]
    
    #for each dimension, optimise and solve for logL = logL_max - 0.5
    op_par = Optimise(LogLikelihood,op_par,func_args,fixed=fixed,verbose=False)
    max_loglik = LogLikelihood(op_par,*func_args)
    err_pos[i] = fsolve(FixedPar_func_offset,op_par[i]+err[i],(max_loglik,LogLikelihood,func_args,fixed,fixed_par)) - op_par[i]
    err_neg[i] = op_par[i] - fsolve(FixedPar_func_offset,op_par[i]-err[i],(max_loglik,LogLikelihood,func_args,fixed,fixed_par)) 
    
    av_err = (np.abs(err_pos)+np.abs(err_neg))/2.
    
    if plot: #make plots of the conditionals with max and limits marked
      par_range = np.linspace(op_par[i]-3*err_neg[i],op_par[i]+3*err_pos[i],plot_samp)
      log_lik = np.zeros(plot_samp)
      temp_par = np.copy(op_par)
      for q,par_val in enumerate(par_range):
        temp_par[i] = par_val
        log_lik[q] = LogLikelihood(temp_par,*func_args)
      pylab.clf()
      pylab.plot(par_range,log_lik)
      pylab.plot(par_range,max_loglik-(par_range-op_par[i])**2/2./av_err[i]**2,'r--')
      pylab.axvline(op_par[i],color='r')
      pylab.axvline(old_op_par[i],color='0.5',ls='--')
      pylab.axvline(op_par[i]+err_pos[i],color='g')
      pylab.axvline(op_par[i]-err_neg[i],color='g')
      pylab.axhline(max_loglik-0.5,color='g',ls='--')
      pylab.axhline(max_loglik,color='r',ls='--')
      pylab.xlabel("p[%s]" % str(i))
      pylab.ylabel("log Posterior")
      #print "mean (+-) = ", op_par[i], err_pos[i], err_neg[i],
      raw_input("")  

  return op_par,(np.abs(err_pos)+np.abs(err_neg))/2.

##########################################################################################
