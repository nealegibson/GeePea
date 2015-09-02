#!/usr/bin/env python

import numpy as np
import GeePea
import pylab

import sys
print sys.path
print GeePea.__file__

#define test data
x = np.linspace(0,1,50)
y = np.sin(2*np.pi*x) + np.random.normal(0,0.1,x.size)

#define the hyperparameters
p = [1,1,0.1]

#define the GP class
GP = GeePea.GP(x,y,p)

GP.optimise() # by default uses the scipy Nelder-Mead simplex fmin
GP.plot()

pylab.xlabel('x')
pylab.ylabel('y')
pylab.draw()

#raw_input()



