#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer
from MyFuncs import LogLikelihood_iid_mf

#define some polynomials
def Model1(p,x):
  return p[0]+p[1]*x
def Model2(p,x):
  return p[0]+p[1]*x+p[2]*x**2
def Model3(p,x):
  return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3

#generate some noisy data
par1 = [0.,0.3,0.01]
par2 = [0.2,0.2,-0.002,0.01]
par3 = [0.2,0.2,-0.002,0.002,0.01]
x = np.linspace(-5,5,100)
y1 = Model1(par1,x) + np.random.normal(0,par1[-1],x.size)
y2 = Model2(par2,x) + np.random.normal(0,par2[-1],x.size)
y3 = Model3(par3,x) + np.random.normal(0,par3[-1],x.size)
y = y3

#plot the data
pylab.errorbar(x,y1,yerr=par1[-1],ls='.')
pylab.errorbar(x,y2,yerr=par2[-1],ls='.')
pylab.errorbar(x,y3,yerr=par3[-1],ls='.')
pylab.plot(x,Model1(par1[:-1],x),'r--')
pylab.plot(x,Model2(par2[:-1],x),'r--')
pylab.plot(x,Model3(par3[:-1],x),'r--')

#MCMC params
chain_len = 25000
conv = 10000
thin=10
no_ch=2
limits = (conv/4,conv,3)

#first try the linear model
gp = par1[:]
ep = [0.001,0.001,0.001]
Infer.MCMC(LogLikelihood_iid_mf,gp,(Model1,x,y),chain_len,ep,n_chains=no_ch,adapt_limits=limits,glob_limits=limits,thin=thin)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
pylab.plot(x,Model1(par[:-1],x),'b')
m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch)
E1,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model1,x,y),m,K,chain_len)
E1,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model1,x,y),m,3.*K,chain_len)

#now try the quadratic model
gp = par2[:]
ep = [0.001,0.001,0.001,0.001]
Infer.MCMC(LogLikelihood_iid_mf,gp,(Model2,x,y),chain_len,ep,n_chains=no_ch,adapt_limits=limits,glob_limits=limits,thin=thin)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
pylab.plot(x,Model2(par[:-1],x),'g')
m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch)
E2,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model2,x,y),m,K,chain_len)
E2,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model2,x,y),m,3.*K,chain_len)

#3rd order...
gp = par3[:]
ep = [0.001,0.001,0.001,0.001,0.001]
Infer.MCMC(LogLikelihood_iid_mf,gp,(Model3,x,y),chain_len,ep,n_chains=no_ch,adapt_limits=limits,glob_limits=limits,thin=thin)
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
pylab.plot(x,Model2(par[:-1],x),'g')
m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch)
E3,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model3,x,y),m,K,chain_len)
E3,par,par_err = Infer.ImportanceSamp(LogLikelihood_iid_mf,(Model3,x,y),m,3.*K,chain_len)

print E1, E2, E3
Infer.BayesFactor(E1,E2)
Infer.BayesFactor(E1,E3,H1='H1',H2='H3')
Infer.BayesFactor(E2,E3,H1='H2',H2='H3')

raw_input()
