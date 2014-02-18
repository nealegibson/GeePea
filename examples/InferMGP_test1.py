#!/usr/bin/env python

import numpy as np
import pylab
from scipy.stats.distributions import gamma,norm as norm_dist

import Infer
from MyFuncs import Transit_aRs

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
gp = [.0003,2.5,10.,.104,0.6,0.2,0.3,1.,0.]
hp = [0.0004,0.1,0.0001]

#create the data set (ie training data)
t = np.linspace(-0.1,0.1,300)
t_pred = np.linspace(-0.12,0.12,1000)
flux = Transit_aRs(lc_pars,t) + 0.0005*np.sin(2*np.pi*40*t) + np.random.normal(0,hp[-1],t.size)

#guess parameter values and guess uncertainties
guess_pars = hp + gp
#err_pars = np.array([0.0004,0.1,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.])

#construct the GP
MyGP = Infer.MGP(flux,np.matrix([t,]).T,p=guess_pars,mf=Transit_aRs,mf_args=t,n_hp=3)
#MyGP.logPrior = lambda p: np.log(norm_dist.pdf(p[6],.10,0.02)).sum()

#make plot of the function
pylab.figure(1)
pylab.plot(MyGP.mf_args,MyGP.t,'k.')
pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'g-')

#get optimised parameters
err_pars = np.array([0.,0.,0.] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.])
MyGP.Optimise(fp=(np.array(err_pars) == 0)*1)
MyGP.Describe()
#pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'r-')

#get optimised parameters and errors using conditional function
#not really working here as the GP hyperparams aren't well constrained
pylab.figure(2)
# err_pars = [4.35780928e-03,2.55132821e+02,1.51694976e-05,8.15399218e-05,\
#    0.00000000e+00,1.74672003e-01,8.06368858e-04,1.58565610e-02,\
#    0.00000000e+00,0.00000000e+00,6.77361776e-03,9.19103594e-04]
# MyGP.pars,err_pars = Infer.ConditionalErrors(MyGP.logPosterior,MyGP.pars,err_pars,(),opt=True,plot=1)

#ML-type 2, ie just set the hyperparameter errors to zero
chain_len = 100000
conv = 40000
thin = 10
no_ch=2
lims = (1000,conv,5)
err_pars = np.array([0.,0.,0.] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.])
Infer.MCMC(MyGP.logPosterior,MyGP._pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin)
MyGP.pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

#do a full GP
err_pars = np.array([0.001,0.001,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.0001])
Infer.MCMC(MyGP.logPosterior,MyGP._pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin)
MyGP.pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

pylab.figure(2)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

pylab.figure(1)
pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'b-')
MyGP.kf_args_pred = np.matrix([t_pred,]).T
MyGP.mf_args_pred = t_pred
Infer.PlotRanges(MyGP.mf_args_pred,*MyGP.Predict())
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVectorFromPrior(),'r--')
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVector(),'g--')

#Importance sample the distribution (assuming Gaussian proposal)
pylab.figure(3)
# m,K = Infer.NormalFromMCMC(conv/thin,n_chains=no_ch,plot=1)
# Infer.ImportanceSamp(MyGP.logPosterior,(),m,2*K,chain_len)

raw_input()
