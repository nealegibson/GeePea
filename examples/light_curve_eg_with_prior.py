#!/usr/bin/env python

import numpy as np
#import pylab
import matplotlib.pyplot as plt
import os

import GeePea
import MyFuncs as MF
import Infer

#define transit parameters
mfp = [0.,2.5,11.,.1,0.6,0.2,0.3,1.,0.] #mean function (transit) parameters
hp = [0.0003,10,0.0003] #hyperparameters
ep = [0.0001,0,0.1,0.001,0.01,0,0,0.0001,0,] #and corresponding error parameters for fitting
ep_hp = [0.0001,1,0.00001] #and hyperparams
mf = MF.Transit_aRs #and mean function, in this case a transit model
#note in principle you could use any python that is called as f(par,*args)
#where par are the fitting variables, some of which can be fixed

#create the data set (ie training data) with some simulated systematics
time = np.linspace(-0.1,0.1,200)
flux = mf(mfp,time) + hp[0]*np.sin(2*np.pi*20.*time) + np.random.normal(0,hp[-1],time.size)

#define the GP kernel
#kf = GeePea.Matern32_inv #this uses an inverse scale parameter
kf = GeePea.ToeplitzMatern32_inv #this assumes a Toeplitz matrix to speed things up, but best don't use unless you understand it...
#often better to fit for log of the length scale and height scale, so could modify this kernel

#define the prior
def logPrior(p,nhp):
  #return -np.inf if restricted prior space
  #this way the full posterior won't be evaluated, which is costly
  #transit parameters
  if np.array(p[:8]<0).any(): return -np.inf
  #hyperparameters
  if np.array(p[-nhp:]<0).any(): return -np.inf
  #limb darkening parameters
  if (p[5] + p[6]) > 1.: return -np.inf #ensure positive surface brightness

  #else calculate the log prior
  log_prior = 0.
  #eg of gamma prior
  #log_prior += np.log(gamma.pdf(p[-nhp+1],1.,0.,1.e2)).sum()
  #eg or normal prior
  #log_prior += np.log(norm_dist.pdf(p[4],b,b_err)).sum()
  return log_prior

#now define the GP
gp = GeePea.GP(time,flux,p=mfp+hp,kf=kf,mf=MF.Transit_aRs,ep=ep+hp)
gp.logPrior = logPrior

#optimise the free parameters - this uses a nelder-mead simplex by default
gp.optimise()

#can also use a global optimiser which is more reliable, but needs a little more effort to set up
#needs a set of tuples defining the lower and upper limit for each parameter
bounds_hp = [(1.e-5,1.e-2),(0.1,250),(0,0.01)]
bounds_mf = [(_p-3*_e,_p+3*_e) for _p,_e in zip(mfp,ep)]
for i,_e in enumerate(ep): #better to set fixed pars to 'None'
  if _e == 0.: bounds_mf[i] = None 
gp.opt_global(bounds=bounds_mf+bounds_hp) #and optimise

#finally make a plot
plt.figure(1)
gp.plot()

#can also run an MCMC by using GP.logPosterior()
lims = (0,10000,4)
Infer.MCMC_N(gp.logPosterior,gp.p,(),20000,gp.ep,N=2,adapt_limits=lims,glob_limits=lims)

#get the parameters and uncertainties from the MCMC
gp.p,gp.ep = Infer.AnalyseChains(10000,n_chains=2)

#and plot the correlations
plt.figure(2)
Infer.PlotCorrelations(10000,n_chains=2,p=np.where(np.array(gp.ep) > 0)[0])
#pylab.savefig('Correlations.png')

#delete the MCMC chains
if os.path.exists('MCMC_chain_1.npy'): os.remove('MCMC_chain_1.npy')
if os.path.exists('MCMC_chain_2.npy'): os.remove('MCMC_chain_2.npy')

raw_input()
