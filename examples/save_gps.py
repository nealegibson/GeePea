#!/usr/bin/env python
"""
Example on how to save GPs using dill (same as pickle but works on instance methods)

"""

import numpy as np
import dill
import GeePea
import os

#individual GP:
x = np.linspace(0,1,200)
y = np.sin(2*np.pi*2*x) + np.random.normal(0,0.2,200)
gp = GeePea.GP(x,y,p=[0.1,0.5,0.1])

#save using dill stream
stream = dill.dumps(gp)
#and recover
gp_copy = dill.loads(stream)

#save using dill file
file = open('test_save.dat','w')
dill.dump(gp,file)
file.close()
#and recover
gp_copy = dill.load(open('test_save.dat'))

#define a list of GPs:
gps = [GeePea.GP(x,y,p=[0.1,0.5,0.1]) for i in range(10)]
#do something to all gps
for gp in gps: gp.opt()
file = open('test_save.dat','w')
dill.dump(gp,file)
file.close()
#and recover
gps_copy = dill.load(open('test_save.dat'))

#also added a save method to the GP class:
gps[5].save('test_save.dat')
gp_copy = dill.load(open('test_save.dat'))

os.remove('test_save.dat')
