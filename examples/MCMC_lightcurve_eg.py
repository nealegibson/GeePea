#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
import MyMCMC
from MyFuncs import Transit_aRs
from MyFuncs import LogLikelihood_iid_mf

#light curve parameters
lc_pars = [2454000.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
wn = 0.0003

#create the data set (ie training data)
time = np.arange(2453999.9,2454000.1,0.001)
flux = Transit_aRs(lc_pars,time) + np.random.normal(0,wn,time.size)

#guess parameter values and guess uncertainties
guess_pars = lc_pars + [wn]
err_pars = [0.00001,0,0.2,0.0003,0.02,0,0,0.,0.,0]

#plot the light curve + guess function
pylab.figure(1)
pylab.errorbar(time,flux,yerr=wn,fmt='.')
pylab.plot(time,Transit_aRs(guess_pars[:-1],time),'r--')

#define MCMC parameters
chain_len = 100000
conv = 20000
thin = 10
no_ch=5

#run the MCMC
Infer.MCMC(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,n_chains=no_ch,adapt_limits=(5000,20000,3),glob_limits=(5000,20000,3),thin=thin)

#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

#plot the chains and correlations
lab = [r'$T_0$',r'$a/R\star$',r'$\rho$',r'$b$']
#pylab.figure(2)
#Infer.PlotChains(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)

#plot fitted function
pylab.figure(1)
pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

#run C version for comparison?
MyMCMC.MCMC(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,n_chains=no_ch,adapt_limits=(5000,20000,5),glob_limits=(5000,20000,5),thin=thin)
par,par_err = MyMCMC.AnalyseChains(conv/thin,n_chains=no_ch)

#remove the chains files
#os.system('rm MCMC_chain_?')
#os.system('rm MCMC_chain_?.npy')

raw_input()
