#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
import MyMCMC
from MyFunctions import Transit_aRs
from MyFunctions import LogLikelihood_iid_mf

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
wn = 0.0003

#create the data set (ie training data)
time = np.arange(-0.1,.1,0.001)
flux = Transit_aRs(lc_pars,time) + np.random.normal(0,wn,time.size)

#guess parameter values and guess uncertainties
guess_pars = lc_pars + [wn]
err_pars = [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.00001,0.00001,0.00001]

#plot the light curve + guess function
pylab.figure(1)
pylab.errorbar(time,flux,yerr=wn,fmt='.')
pylab.plot(time,Transit_aRs(guess_pars[:-1],time),'r--')

#first optimise the function
#guess_pars = Infer.Optimise(LogLikelihood_iid_mf,guess_pars[:],(Transit_aRs,time,flux),fixed=(np.array(err_pars) == 0)*1)

#find the conditional errors to seed an MCMC
pylab.figure(2)
p,e = Infer.ConditionalErrors(LogLikelihood_iid_mf,guess_pars,err_pars,(Transit_aRs,time,flux),plot=1)

#run the MCMC
#define MCMC parameters
chain_len = 100000
conv = 50000
thin = 10
no_ch=2
adapt_lims = (0,0,0)
glob_lims = (2000,100000,4)
Infer.MCMC(LogLikelihood_iid_mf,p,(Transit_aRs,time,flux),chain_len,e,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
