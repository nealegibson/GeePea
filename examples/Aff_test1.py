#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
import MyMCMC
from MyFunctions import Transit_aRs
from MyFunctions import LogLikelihood_iid_mf

def Posterior(p,*args):
  if p[-1] < 0.: return -np.inf
  if np.any(p < 0.): return -np.inf
  return LogLikelihood_iid_mf(p,*args)

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
wn = 0.0003

#create the data set (ie training data)
time = np.linspace(-0.1,0.1,300)
flux = Transit_aRs(lc_pars,time) + np.random.normal(0,wn,time.size)

#guess parameter values and guess uncertainties
guess_pars = lc_pars + [wn]
err_pars = [0.0001,0,0.05,0.003,0.002,0.0,0.0,0.01,0.0001,0.00010]

#plot the light curve + guess function
pylab.figure(1)
pylab.errorbar(time,flux,yerr=wn,fmt='.')
pylab.plot(time,Transit_aRs(guess_pars[:-1],time),'r--')

#first optimise the function
guess_pars = Infer.Optimise(Posterior,guess_pars[:],(Transit_aRs,time,flux),fixed=(np.array(err_pars) == 0)*1,method='BFGS')

#run a normal MCMC
chain_len = 60000
conv = 30000
thin = 10
no_ch=3
adapt_lims = (2000,conv,10)
glob_lims = (2000,conv,10)
# Infer.MCMC(Posterior,guess_pars[:],(Transit_aRs,time,flux),chain_len,err_pars,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
# #Get parameters values/errors from chains
# par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
# bf_par = Infer.GetBestFit(n_chains=no_ch)
# print "Best Fit log p =", LogLikelihood_iid_mf(bf_par,Transit_aRs,time,flux)
# pylab.figure(3)
# Infer.PlotCorrelations(conv/thin,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#run an affine inv MCMC
n = 300
chain_len = chain_len/n
conv = conv/n
no_ch=3
Infer.AffInvMCMC(Posterior,guess_pars[:],(Transit_aRs,time,flux),n,chain_len,err_pars,n_chains=no_ch)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv*n,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", LogLikelihood_iid_mf(bf_par,Transit_aRs,time,flux)
pylab.figure(4)
Infer.PlotCorrelations(conv*n,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#plot the chains and correlations
#lab = [r'$T_0$',r'$a/R\star$',r'$\rho$',r'$b$']
#pylab.figure(2)
#Infer.PlotChains(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)

#plot fitted function
pylab.figure(1)
pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

raw_input()
