#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
from MyFuncs import Transit_aRs
from MyFuncs import LogLikelihood_iid_mf

#define a test function: 2D Rosenbrock function (inverted)
def InvRosenbrock(p):
  R = (1.-p[0])**2. + 100.*(p[1]-p[0]**2.)**2.  
  return -R

#multivariate - sum of p/2 coupled Rosenbrock functions
def InvNRosenbrock(p):
  R = 0.
  for i in range(len(p)/2):
    R += 100.*(p[2*i]**2-p[2*i+1])**2. + (p[2*i]-1.)**2.
  return -0.4*R

gp = [1.,1.,1.,1.,1.,1.,1.,1.]
ep = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

gp = np.array(gp) + np.random.normal(0,1,len(gp))

#first optimise the function
gp = Infer.Optimise(InvNRosenbrock,gp[:],(),fixed=(np.array(ep) == 0)*1)

#run a normal MCMC
chain_len = 50000
conv = 2000
thin = 1
no_ch=3
adapt_lims = (200,conv,10)
glob_lims = (200,conv,10)
Infer.MCMC(InvNRosenbrock,gp[:],(),chain_len,ep,n_chains=no_ch,adapt_limits=adapt_lims,glob_limits=glob_lims,thin=thin)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", InvNRosenbrock(bf_par,)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

#run an affine inv MCMC
n = 200
chain_len = chain_len/n
conv = conv
no_ch=3
Infer.AffInvMCMC(InvNRosenbrock,gp[:],(),n,chain_len,ep,n_chains=no_ch)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", InvNRosenbrock(bf_par,)
pylab.figure(4)
Infer.PlotCorrelations(conv*n,n_chains=no_ch,p=np.where(np.array(par_err)>0.)[0])

# 
# 
# #plot the chains and correlations
# #lab = [r'$T_0$',r'$a/R\star$',r'$\rho$',r'$b$']
# #pylab.figure(2)
# #Infer.PlotChains(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)
# 
# #plot fitted function
# pylab.figure(1)
# pylab.plot(time,Transit_aRs(par[:-1],time),'g-')

raw_input()
