import GeePea
import numpy as np

my_mean_func = lambda p,x: p[0] + p[1] * x

x = np.linspace(0,1,50)
y = my_mean_func([1.,3.],x) + np.sin(2*np.pi*x) + np.random.normal(0,0.1,x.size)

mfp = [0.8,2.] # mean function parameters
hp = [1.,1.,0.1] # kernel hyperparameters (sq exponential takes 3 parameters for 1D input)

gp = GeePea.GP(x,y,p=mfp+hp,mf=my_mean_func)
gp.optimise()
gp.plot()