#!/usr/bin/env python

import numpy as np
import pylab
import os

import Infer

def LogNormalDist(p,m,invK,logdetK):
  #mean subtract the vector
  r = np.matrix(np.array(p-m).flatten()).T
  #and calculate the log prob
  logP = -0.5 * r.T * invK * r - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)
  return np.float(logP)

#create modle parameters
m = np.array([1,2,3])
K = np.diag([0.5,0.5,0.5])
invK = np.mat(K).I
sign,logdetK = np.linalg.slogdet(K) #calcualte detK

#
gp = [0.7,1.7,3.2]
ep = [0.01,0.01,0.01]

print "means =", m
print "errs =", np.sqrt(np.diag(K))

#define MCMC parameters
chain_len = 40000
conv = 20000
thin = 10
no_ch=2

#run the MCMC
Infer.MCMC(LogNormalDist,gp,(m,invK,logdetK),chain_len,ep,n_chains=no_ch,adapt_limits=(5000,20000,5),glob_limits=(5000,20000,5),thin=thin)
#Get parameters values/errors from chains
par,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
bf_par = Infer.GetBestFit(n_chains=no_ch)
print "Best Fit log p =", LogNormalDist(bf_par,m,invK,logdetK)

#plot the chains and correlations
#pylab.figure(2)
#Infer.PlotChains(conv/thin,n_chains=no_ch,p=[0,2,3,4],labels=lab)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

############ Importance sampling stage #################
#get mean and covariance from the MCMC chains
m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch)
#or use 'perfect' mean and K
#m = np.array([1,2,3])
#K = np.diag([0.5,0.5,0.5])

Infer.ImportanceSamp(LogNormalDist,(m,invK,logdetK),m,K,chain_len)
#run twice to test consistency
Infer.ImportanceSamp(LogNormalDist,(m,invK,logdetK),m,K,chain_len)

raw_input()
