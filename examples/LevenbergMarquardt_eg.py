#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
import MyFuncs as MF

#create transit parameters (pars, errors, guess)
tpar = np.array([0.0,1.0,10.,0.1,0.1,0.2,0.2,1.0,0.0,0.0003])
epar = np.array([0.0001,0.0,0.02,0.01,0.02,0.0,0.0,0.0001,0.00001,0.0000])
gpar = np.array([0.001,1.0,12.,0.13,0.2,0.2,0.2,1.001,0.00001,0.0003])

#use fewer variable parameters? NB The evidence approximation will be better the closer to Gaussian
#the posterior is
#gpar = tpar
#epar = np.array([0.000,0.0,0.0,0.01,0.0,0.0,0.0,0.0001,0.00001,0.0000])

#create arrays for bound and fixed parameters (optional)
fixed = (epar == 0) * 1
bounds = np.array([(None,None) for i in range(len(tpar))])
bounds[4][0] = 0.

#create data (time and flux)
t = np.linspace(-0.05,0.05,1000)
f = MF.Transit_aRs(tpar[:-1],t) + np.random.normal(0,tpar[-1],t.size)

#perform LM fit to the data
p,pe,wn,K,logE = Infer.LevMar(MF.Transit_aRs,gpar[:-1],(t,),f,fixed=fixed[:-1],bounds=None)

#get residuals
resid = f - MF.Transit_aRs(p,t)

#compare with MCMC fit - use LM values as inputs 
lims = (0,4000,4)
MCMC_p = list(p) + [wn,]
MCMC_pe = list(pe) + [epar[-1],]
Infer.MCMC_N(MF.LogLikelihood_iid_mf,MCMC_p,(MF.Transit_aRs,t,f),10000,MCMC_pe,adapt_limits=lims,glob_limits=lims,N=2)
MCMC_p,MCMC_pe = Infer.AnalyseChains(lims[1],n_chains=2,N_obs=t.size)
os.remove('MCMC_chain_1.npy');os.remove('MCMC_chain_2.npy')

#plot the data
pylab.plot(t,f,'k.')
pylab.plot(t,MF.Transit_aRs(gpar[:-1],t),'g-')
pylab.plot(t,MF.Transit_aRs(p,t),'r-')
pylab.plot(t,resid+1.0-1.5*np.ptp(MF.Transit_aRs(p,t)),'k.')
pylab.axhline(1.0-1.5*np.ptp(MF.Transit_aRs(p,t)),color='r')
pylab.plot(t,MF.Transit_aRs(MCMC_p[:-1],t),'b-')

raw_input()