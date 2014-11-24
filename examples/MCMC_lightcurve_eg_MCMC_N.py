#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
#import MyMCMC

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
err_pars = [0.00001,0.,0.2,0.0003,0.02,0.0,0.0,0.0001,0.0001,0.00001]

#plot the light curve + guess function
pylab.figure(1)
pylab.errorbar(time,flux,yerr=wn,fmt='.')
pylab.plot(time,Transit_aRs(guess_pars[:-1],time),'r--')

#define MCMC parameters
chain_len = 5000
conv = 2000
thin = 1
no_ch=5
lims = (2000,10000,4)

#run the MCMC
# MCMCParallel.MCMCParallel2(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,adapt_limits=lims,glob_limits=lims,thin=thin)
# par,par_err = Infer.AnalyseChains(chain_len,chain_filenames=['MCMCpar_chain1','MCMCpar_chain2'])
# pylab.figure(2)
# Infer.PlotCorrelations(chain_len,chain_filenames=['MCMCpar_chain1','MCMCpar_chain2'])

#test the N parallel chain code
chain_len = 11000
conv = 10000
thin = 1
lims = (0,conv,4)
ext_len = 1000
max_ext = 50
N = 4
import time as timer
start = timer.time()
Infer.MCMC_N(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,N=N,adapt_limits=lims,glob_limits=lims,thin=thin,ext_len=ext_len,max_ext=max_ext)
ts = timer.time()-start
st1 = "t = %dm %.2fs" % (ts // 60., ts % 60.)
#MCMCParallelCPU.MCMCParallel(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,N=N,adapt_limits=lims,glob_limits=lims,thin=thin,ext_len=ext_len,max_ext=100)
par,par_err = Infer.AnalyseChains(conv,n_chains=N)
pylab.figure(2)
Infer.PlotCorrelations(conv,n_chains=N)

#check against normal version
# chain_len = 10000
# conv = 4000
# lims = (0,conv,4)
start = timer.time()
Infer.MCMC(LogLikelihood_iid_mf,guess_pars,(Transit_aRs,time,flux),chain_len,err_pars,adapt_limits=lims,glob_limits=lims,thin=thin,n_chains=N)
ts = timer.time()-start
st2 = "t = %dm %.2fs" % (ts // 60., ts % 60.)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=N)
pylab.figure(3)
Infer.PlotCorrelations(conv,n_chains=N)

#plot fitted function
pylab.figure(1)
pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

print st1
print st2

raw_input()
