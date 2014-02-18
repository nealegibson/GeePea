#!/usr/bin/env python

import numpy as np
import pylab
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys

import Infer

#define log likelihood (i.i.d.) function (after subtracting mean function)
def LogLikelihood_iid_mf(p,func,x,y,sig=1.):
  r = y - func(p[:-1],x)
  N = r.size
  logP = - (1./2.) * ( r**2 / (p[-1]*sig)**2 ).sum() - np.log(sig).sum() - N*np.log(p[-1]) - N/2.*np.log(2*np.pi)
  return logP

def f(p,x): #sin function to fit
  return p[0]*(np.sinc(p[1]*x-4))**2 + p[2]

lf = LogLikelihood_iid_mf

#actual parameter values
p = [3.,3.,0.3,0.1]

#guess parameter values and guess uncertainties
pg = [2.,3.3,0.1,0.3]
dp = [0.2, 0.028, 0.0008, 0.026]
#dp_log = [0.003213, 0.000396, 0.004990, 0.014538]

#make noisy data to fit
x = np.r_[0:8:0.01]
y = f(p[:3],x) + np.random.normal(0,p[3],x.size)

#chain_files = ("MCMC_chain_1","MCMC_chain_2","MCMC_chain_3")
#chain_files = "test_file"
chain_len = 25000
conv = 20000
thin=1
no_ch=5
limits = (conv/5,conv,2)

#run the ballpark thinger
# pl = [0.001,0.001,0.001,0.01]
# pu = [20.,10.0,0.9,1.]
# MyMCMC.Ballpark(lf,pl,pu,(f,x,y),5000,filename="testfile",log_proposal=[1,1,1,1])
# pg = MyMCMC.ExtractBallpark(filename="testfile")

#plot test function
pylab.figure(1)
pylab.plot(x,y,'k.')
pylab.plot(x,f(p[:3],x),'r-')
pylab.plot(x,f(pg[:3],x),'g--') #plot guess function

#run the MCMC (using C version)
MyMCMC.MCMC(lf,pg,(f,x,y),chain_len,dp,n_chains=no_ch,glob_limits=limits,adapt_limits=limits)
#Get parameters values/errors from chains
par,par_err = MyMCMC.AnalyseChains(conv,n_chains=no_ch)
print

#run the PyMCMC
Infer.MCMC(lf,pg,(f,x,y),chain_len,dp,n_chains=no_ch,glob_limits=limits,adapt_limits=limits,thin=thin)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
print

#plot fitted function
pylab.figure(1)
pylab.plot(x,f(par[:3],x),'b-')
# pylab.figure(2)
# Infer.PlotChains(conv/thin,n_chains=no_ch)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch,n_samples=1000)

raw_input()
