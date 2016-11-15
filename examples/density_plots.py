#!/usr/bin/env python

import numpy as np
import pylab
import os
import matplotlib as mpl

import GeePea
import MyFuncs as MF
import Infer

#create example with some noise/systematics
tpar = [0,3.0,10,0.1,0.2,0.2,0.2,1.0,0.0]
time = np.linspace(-0.1,0.1,300)
flux = MF.Transit_aRs(tpar,time) + 0.001*np.sin(2*np.pi*40*time) + np.random.normal(0.,0.0005,time.size)

#construct the GP
gp = GeePea.GP(time,flux,p=tpar+[0.1,0.01,0.001],mf=MF.Transit_aRs)
gp.opt() #optimise

#run quick MCMC to test predictions:
ch_len = 10000
lims = (0,5000,10)
epar = [0,0,0,0.001,0,0,0,0,0,] + [0.001,0.01,0.001]
Infer.MCMC_N(gp.logPosterior,gp.p,(),ch_len,epar,adapt_limits=lims,glob_limits=lims,chain_filenames=['test_chain'])
p,perr = Infer.AnalyseChains(lims[1],chain_filenames=['test_chain'])
X = Infer.GetSamples(5000,100,chain_filenames=['test_chain']) #get samples from the chains
os.remove('test_chain.npy')

#standard plot
pylab.figure()
gp.plot()
#pylab.savefig('test.pdf')

#density plot for single parameter set
pylab.figure()
f,ferr = gp.predict()
GeePea.PlotDensity(time,f,ferr)
pylab.plot(time,flux,'ro',ms=3)

#density plot for sample of params
pylab.figure()
f,ferr = gp.predictSample(p=X[:])
GeePea.PlotDensity(time,f,ferr,supersamp=10)
pylab.plot(time,flux,'ro',ms=3)
pylab.colorbar()

#density plot from random draws from sample
pylab.figure()
f,ferr = gp.predictDraws(p=X[:])
GeePea.PlotDensity(time,f,ferr,supersamp=5)
pylab.plot(time,flux,'ro',ms=3)

#eg without white noise and supersampling
pylab.figure()
f,ferr = gp.predictSample(p=X[:],wn=0)
GeePea.PlotDensity(time,f,ferr,supersamp=10)
pylab.plot(time,flux,'ro',ms=3)

##########################################################################################
#make plot with multiple GP draws - hard to take into account the white noise though
#just plot lots of random vectors with a sensible alpha
pylab.figure()
V = gp.getRandomVectors(p=X)
plot(time,V[:].T,'k-',alpha=0.02,lw=1)
pylab.plot(time,flux,'o',ms=4,mec='k',mfc='r')

##########################################################################################
#steal a current colormap and add alpha to it (this is important for overplotting transits)
#this is less important since I've masked the PlotDensity map, but will still leave it
#here for future ref
cmap = mpl.cm.Greys
cols = cmap(np.arange(cmap.N)) #get colors from exisiting colormap
cols[:,-1] = np.linspace(0, 1, cmap.N) #set alpha from 0 to 1, ie transparent for low values
my_cmap1 = mpl.colors.ListedColormap(cols)
my_cmap1._init()

#create a custom colormap
#my_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','black'],256)
my_cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','red'],256)
cmap = mpl.cm.Reds
cols = my_cmap2(np.arange(256))
cols[:,-1] = np.linspace(0, 1, my_cmap1.N)
my_cmap2 = mpl.colors.ListedColormap(cols)
my_cmap2._init()

f,ferr = gp.predict()
I = GeePea.PlotDensity(time,f,ferr,supersamp=None,cmap=my_cmap1)
pylab.colorbar()
I = GeePea.PlotDensity(time,f-0.01,ferr,supersamp=None,cmap=my_cmap2)
pylab.colorbar()
#pylab.savefig('test.png')
#pylab.savefig('test.pdf')

##########################################################################################

pylab.show()
pylab.draw()
raw_input()

