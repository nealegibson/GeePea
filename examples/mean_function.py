#!/usr/bin/env python

import GeePea
import numpy as np

#first define mean function in correct format
my_mean_func = lambda p,x: p[0] + p[1] * x

#create test data
x = np.linspace(0,1,50)
y = my_mean_func([1.,3.],x) + np.sin(2*np.pi*x) + np.random.normal(0,0.1,x.size)

#define mean function parameters and hyperparameters
mfp = [0.8,2.]
hp = [1.,1.,0.1] # kernel hyperparameters (sq exponential takes 3 parameters for 1D input)

#define the GP
gp = GeePea.GP(x,y,p=mfp+hp,mf=my_mean_func)

#print out the GP attributes
gp.describe()

#optimise and plot
gp.optimise()
gp.plot()

raw_input()
