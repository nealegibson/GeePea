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

#define MCMC parameters
chain_len = 20000
conv = 10000
thin = 10
no_ch=2
adapt_lims = (2000,10000,3)
glob_lims = (2000,20000,10)

#first optimise the function and get parameter errors from the conditionals
guess_pars,err_pars = Infer.ConditionalErrors(LogLikelihood_iid_mf,guess_pars,err_pars,(Transit_aRs,time,flux),plot=0,opt=True)
Infer.MCMC(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

#plot the chains and correlations
#pylab.figure(2)
#Infer.PlotChains(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#importance sampling to get the evidence and re-estimate the mean/std of each parameter
pylab.figure(4)
m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch,plot=1) #get covariance matrix of MCMC chains
Infer.ImportanceSamp(LogLikelihood_iid_mf,(Transit_aRs,time,flux),m,2*K,10000) #importance sample

#plot fitted function
pylab.figure(1)
pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

raw_input()
