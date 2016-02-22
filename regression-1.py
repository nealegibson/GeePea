import GeePea
import numpy as np

#create test data
x = [-1.50,-1.0,-0.75,-0.4,-0.25,0.00]
y = [-1.6,-1.1,-0.5,0.25,0.5,0.8]

#define mean function parameters and hyperparameters
hp = [0.,1.,0.3] # kernel hyperparameters (sq exponential takes 3 parameters for 1D input)
fp = [0,0,1]

#define the GP
gp = GeePea.GP(x,y,p=hp,fp=fp)

#optimise and plot
gp.optimise()
gp.plot()