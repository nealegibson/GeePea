#!/usr/bin/env python

import numpy as np
import pylab
import os
import scipy.signal as SS

import GeePea
import MyFuncs as MF
import Infer

#define transit parameters
mfp = [0.,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
#hp = [0.0003,0.1,0.1,0.0003]
hp = [0.0003,10,0.0003,10,0.0003]
ep = [0.00004,0,0.1,0.001,0.01,0,0,0.002,0,0.0001,0.1,0.0001,0.1,0.00001]

#create the data set (ie training data) with some simulated systematics
time = np.linspace(-0.1,0.1,200)
flux = MF.Transit_aRs(mfp,time) + np.random.normal(0,hp[-1],time.size)

#create X array
x0 = time
x0 = (x0-x0.mean())/x0.std()
x1 = SS.sawtooth(2*np.pi*time*15.)
x1 = (x1-x1.mean())/x1.std()
X = np.array([x0,x1]).T

flux += 0.0003*np.sin(2*np.pi*10.*time)
flux += x1 * 0.0004

#define the GP
gp = GeePea.GP(X,flux,xmf=time,p=mfp+hp,mf=MF.Transit_aRs,ep=ep,kf=GeePea.SqExponentialSum)
def logPrior(p,n):
  if p[-n]<0: return -np.inf
  if p[-n+1]<0: return -np.inf
  if p[-n+2]<0: return -np.inf
  if p[-n+3]<0: return -np.inf
  if p[-n+4]<0: return -np.inf
  return 0
gp.logPrior = logPrior

#optimise the free parameters
gp.optimise()
gp.plot()

#can also run an MCMC by using GP.logPosterior()
ch = 10000
conv = 0.4*ch
lims = (0,conv,10)
Infer.MCMC_N(gp.logPosterior,gp.p,(),ch,gp.ep,N=2,adapt_limits=lims,glob_limits=lims)
#Infer.AffInvMCMC(gp.logPosterior,gp.p,(),500,ch/500,gp.ep*0.01,n_chains=2)
pylab.figure(2)
Infer.PlotCorrelations(conv,n_chains=2,p=np.where(np.array(gp.ep) > 0)[0])
#pylab.savefig('Correlations.png')

#get the parameters and uncertainties from the MCMC
gp.p,gp.ep = Infer.AnalyseChains(conv,n_chains=2)

#and plot
pylab.figure(1)
gp.plot()

#delete the MCMC chains
# if os.path.exists('MCMC_chain_1.npy'): os.remove('MCMC_chain_1.npy')
# if os.path.exists('MCMC_chain_2.npy'): os.remove('MCMC_chain_2.npy')

#raw_input()
