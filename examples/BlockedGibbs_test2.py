#!/usr/bin/env python

import numpy as np
import pylab
from scipy.stats.distributions import gamma,norm as norm_dist
import IPython

import Infer
from MyFuncs import Transit_aRs

#light curve parameters
lc_pars = [.0,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
gp =      [.01,2.5,10.,.104,0.6,0.2,0.3,1.,0.]
hp = [0.00078,0.008,0.0003]

#create the data set (ie training data)
t = np.linspace(-0.1,0.1,300)
t_pred = np.linspace(-0.12,0.12,1000)
flux = Transit_aRs(lc_pars,t) + 0.0003*np.sin(2*np.pi*40*t) + np.random.normal(0,hp[-1],t.size)

#guess parameter values and guess uncertainties
guess_pars = hp + gp
err_pars = np.array([0.00005,0.001,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.001])

#construct the GP
MyGP = Infer.GP(flux,np.matrix([t,]).T,p=guess_pars,mf=Transit_aRs,mf_args=t,n_hp=3,n_store=5)
#MyGP.logPrior = lambda p: np.log(norm_dist.pdf(p[6],.10,0.02)).sum()

#make plot of the function
pylab.figure(1)
pylab.plot(MyGP.mf_args,MyGP.t,'k.')
pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'g-')

#get optimised parameters
MyGP.Pars(Infer.Optimise(MyGP.logPosterior,guess_pars[:],(),fixed=(np.array(err_pars) == 0)*1))
pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'r-')

low = np.array(guess_pars) - np.array(err_pars)
upp = np.array(guess_pars) + np.array(err_pars)

#Infer.PlotConditionals(MyGP.logPosterior,MyGP._pars,err_pars,low,upp)
#IPython.embed(banner1='provide new errors for conditional function?')
err_pars = np.array([0.0002,0.0004,0.000015] + [0.0002,0,0.05,0.005,0.02,0.,0.,0.001,0.0006])

pylab.figure(10)
guess_pars,err_pars = Infer.ConditionalErrors(MyGP.logPosterior,MyGP._pars,err_pars,([]),plot=1,opt=0)

#ML-type 2, ie just set the hyperparameter errors to zero
chain_len = 20000
conv = 10000
thin = 10
no_ch=2
lims = (1000,conv,5)
# err_pars = np.array([0.,0.,0.] + [0.00001,0,0.2,0.0003,0.02,0.0,0.0,0.001,0.])
# Infer.MCMC(MyGP.logPosterior,MyGP.pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin)
# MyGP.pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)

#do a full GP
#err_pars = np.array([0.001,0.001,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.,0.,0.001,0.0001])
Infer.MCMC(MyGP.logPosterior,MyGP._pars,(),chain_len,err_pars,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin)
pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
pylab.figure(2)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

#do a Blocked Gibbs MCMC
chain_len = 10000
conv = 5000
thin = 10
no_ch=2
lims = (1000,conv,5)
#err_pars = np.array([0.001,0.001,0.0001] + [0.00001,0,0.2,0.0003,0.02,0.,0.,0.001,0.0001])
#gibbs_in = [1,1,1,2,0,2,2,2,2,2,2,2]
gibbs_in = [1,1,1,2,0,3,3,3,0,0,2,2]
Infer.BGMCMC(MyGP.logPosterior,MyGP._pars,(),chain_len,err_pars,gibbs_in,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin)
pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
pylab.figure(3)
Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

#and repeat with full gibbs on the transit parameters
#gibbs_in = [1,1,1,2,0,2,2,2,0,0,2,2]
# gibbs_in = [1,1,1,2,0,3,4,5,6,7,8,9]
# acc = [0.3,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44] #target acceptance array
# Infer.BGMCMC(MyGP.logPosterior,MyGP._pars,(),chain_len,err_pars,gibbs_in,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin,acc=acc)
# pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
# pylab.figure(4)
# Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

#and repeat with orthogonal steps
# Infer.BGMCMC(MyGP.logPosterior,MyGP.pars,(),chain_len,err_pars,gibbs_in,n_chains=no_ch,adapt_limits=lims,glob_limits=lims,thin=thin,orth=1)
# pars,par_err = Infer.AnalyseChains(conv/thin,n_chains=no_ch)
# pylab.figure(4)
# Infer.PlotCorrelations(conv/thin,n_chains=no_ch)

MyGP._pars = pars
pylab.figure(1)
pylab.plot(MyGP.mf_args,MyGP.MeanFunction(),'b-')
MyGP.kf_args_pred = np.matrix([t_pred,]).T
MyGP.mf_args_pred = t_pred
Infer.PlotRanges(MyGP.mf_args_pred,*MyGP.Predict())
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVectorFromPrior(),'r--')
pylab.plot(MyGP.mf_args_pred,MyGP.GetRandomVector(),'g--')

raw_input()
