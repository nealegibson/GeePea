#!/usr/bin/env python

import GeePea
import numpy as np
import pylab

x = np.linspace(0,1,50)
y = np.sin(2*np.pi*x) + np.random.normal(0,0.1,x.size)

p = [1,1,0.1]

GP = GeePea.GP(x,y,p)

GP.optimise()

GP.plot()

pylab.xlabel('x')
pylab.ylabel('y')

raw_input()
