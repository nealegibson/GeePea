#!/usr/bin/env python

import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time

import pylab

import Infer
from MyFuncs import Transit_aRs

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
hp = [0.0003,0.01,0.0003]

#create the data set (ie training data)
t = np.linspace(-0.1,0.1,300)
t_pred = np.linspace(-0.12,0.12,1000)
flux = Transit_aRs(lc_pars,t) + np.random.normal(0,hp[-1],t.size)

#guess parameter values and guess uncertainties
guess_pars = hp + lc_pars
err_pars = np.array([0.00001,1,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.0001])

X = np.matrix([t,]).T
X_pred = np.matrix([t_pred,]).T

MyGP = Infer.GP(flux,X,p=guess_pars,mf=Transit_aRs,mf_args=t,n_hp=3)

#make plot of the function
pylab.figure(1)
pylab.plot(MyGP.mf_args,MyGP.t,'k.')
pylab.plot(MyGP.mf_args,MyGP.mf(lc_pars,MyGP.mf_args),'g-')

start = time.time() 
for i in range(100):
  gp = np.array(guess_pars) + np.random.normal(0,1) * err_pars
#  print gp
  lik = MyGP.logPosterior(gp)
#  print lik
print " t = %.2f s" % (time.time()-start)

MyGP.pars = Infer.Optimise(MyGP.logPosterior,guess_pars[:],(),fixed=(np.array(err_pars) == 0)*1)

#define MCMC parameters
chain_len = 10000
conv = 2000
thin = 10
no_ch=2

#full GP
Infer.MCMC(MyGP.logPosterior,guess_pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=(500,2000,3),glob_limits=(500,2000,3),thin=thin)
MyGP.pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

#ML-type 2, ie just set the hyperparameter errors to zero!
err_pars = np.array([0,0,0] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.0001])
Infer.MCMC(MyGP.logPosterior,guess_pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=(500,2000,3),glob_limits=(500,2000,3),thin=thin)
MyGP.pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

MyGP.Describe()
MyGP.kf_args_pred = X_pred
MyGP.mf_args_pred = t_pred
MyGP.pars[0:2] = [0.0001,50000]
MyGP.Describe()

pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVector(),'b-')
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVectorFromPrior(),'r')
MyGP.pars[0:3] = [0.0005,50000,0.0009]
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVector(),'b-')
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVectorFromPrior(),'r')

f_pred,f_pred_err = MyGP.Predict()
Infer.PlotRanges(t_pred,*MyGP.Predict())
f_pred,f_pred_err = MyGP.Predict()
pylab.plot(t_pred,f_pred,'g--')

raw_input()
