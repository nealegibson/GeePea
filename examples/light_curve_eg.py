#!/usr/bin/env python

import numpy as np
import pylab
import os

import GeePea
import MyFuncs as MF
import Infer

#define transit parameters
mfp = [0.,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
hp = [0.0003,0.1,0.0003]
ep = [0.0001,0,0.1,0.001,0.01,0,0,0.0001,0,0.0001,0.001,0.00001]

#create the data set (ie training data) with some simulated systematics
time = np.linspace(-0.1,0.1,200)
flux = MF.Transit_aRs(mfp,time) + hp[0]*np.sin(2*np.pi*10.*time) + np.random.normal(0,hp[-1],time.size)

#define the GP
#GP = GeePea.GP(time,flux,p=mfp+hp,mf=MF.Transit_aRs,ep=ep) #using normal SqExponential kernel
GP = GeePea.GP(time,flux,p=mfp+hp,kf=GeePea.ToeplitzSqExponential,mf=MF.Transit_aRs,ep=ep)

#optimise the free parameters
GP.optimise()

#and plot
pylab.figure(1)
GP.plot()

#can also run an MCMC by using GP.logPosterior()
lims = (0,10000,4)
Infer.MCMC_N(GP.logPosterior,GP.p,(),20000,GP.ep,N=2,adapt_limits=lims,glob_limits=lims)

#get the parameters and uncertainties from the MCMC
GP.p,GP.ep = Infer.AnalyseChains(10000,n_chains=2)

#and plot the correlations
pylab.figure(2)
Infer.PlotCorrelations(10000,n_chains=2)

#delete the MCMC chains
if os.path.exists('MCMC_chain_1.npy'): os.remove('MCMC_chain_1.npy')
if os.path.exists('MCMC_chain_2.npy'): os.remove('MCMC_chain_2.npy')

raw_input()
