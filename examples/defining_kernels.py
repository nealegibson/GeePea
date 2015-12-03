#!/usr/bin/env python

import GeePea
import numpy as np

# define mean function in correct format
my_mean_func = lambda p,x: p[0] + p[1] * x

#define a new kernel function
def MySqExponential(X,Y,theta,white_noise=False):
  """
  Just copied from GeePea.SqExponential, using log height scale par
  """
  
  #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
  D2 = GeePea.EuclideanDist2(X,Y,v=1./(np.array(theta[1:-1])))#
  
  #Calculate covariance matrix
  K = np.exp(2*theta[0]) * np.exp( - 0.5 * D2 )

  #Add white noise
  if white_noise is True: K += np.identity(X[:,0].size) * (theta[-1]**2)

  return np.matrix(K)
#add some attributes to the kernel
MySqExponential.n_par = lambda D: D+2
MySqExponential.kernel_type = "Full"

#create test data
x = np.linspace(0,1,50)
y = my_mean_func([1.,3.],x) + np.sin(2*np.pi*x) + np.random.normal(0,0.1,x.size)

#define mean function parameters and hyperparameters
mfp = [0.8,2.]
hp = [0.,1.,0.1] # kernel hyperparameters (sq exponential takes 3 parameters for 1D input)

#define the GP
gp = GeePea.GP(x,y,p=mfp+hp,kf=MySqExponential,mf=my_mean_func)

#print out the GP attributes
gp.describe()

#optimise and plot
gp.optimise()
gp.plot()

raw_input()
