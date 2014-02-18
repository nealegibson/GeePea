#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
from MyFuncs import Transit_aRs
from MyFuncs import LogLikelihood_iid_mf

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
wn = 0.0003

#create the data set (ie training data)
time = np.arange(-0.1,0.1,0.001)
flux = Transit_aRs(lc_pars,time) + np.random.normal(0,wn,time.size)

#guess parameter values and guess uncertainties
guess_pars = lc_pars + [wn]
err_pars = [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.0001,0.0001]

#plot the light curve + guess function
pylab.figure(1)
pylab.errorbar(time,flux,yerr=wn,fmt='.')
pylab.plot(time,Transit_aRs(guess_pars[:-1],time),'r--')

#first optimise the function
guess_pars = Infer.Optimise(LogLikelihood_iid_mf,guess_pars[:],(Transit_aRs,time,flux),fixed=(np.array(err_pars) == 0)*1)

#run a standard MCMC
chain_len = 40000
conv = 10000
thin = 10
no_ch=2
adapt_lims = (2000,conv,5)
glob_lims = (2000,conv,5)
glob_lims = (0,0,0)
Infer.MCMC(LogLikelihood_iid_mf,guess_pars[:],(Transit_aRs,time,flux),chain_len,err_pars,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", LogLikelihood_iid_mf(bf_par,Transit_aRs,time,flux)
pylab.figure(2)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#run a Blocked Gibbs MCMC
gibbs_in = [1,0,2,2,2,0,0,1,1,1]
no_steps = max(gibbs_in)
chain_len = 20000
conv = 10000/2
thin = 10
no_ch=2
adapt_lims = (1000,conv,5)
glob_lims = (1000,conv,5)
glob_lims = (0,0,0)
Infer.BGMCMC(LogLikelihood_iid_mf,guess_pars[:],(Transit_aRs,time,flux),chain_len,err_pars,gibbs_in,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", LogLikelihood_iid_mf(bf_par,Transit_aRs,time,flux)
pylab.figure(3)
Infer.PlotCorrelations(conv/no_steps/thin,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#plot fitted function
pylab.figure(1)
pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

raw_input()
