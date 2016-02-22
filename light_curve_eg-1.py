import GeePea
import numpy as np
import MyFuncs as MF
import pylab

#define transit parameters
mfp = [0.,2.5,11.,.1,0.6,0.2,0.3,1.,0.]
hp = [0.0003,0.1,0.0003]
ep = [0.0001,0,0.1,0.001,0.01,0,0,0.0001,0,0.0001,0.001,0.00001]

#create the data set (ie training data) with some simulated systematics
time = np.arange(-0.1,0.1,0.001)
time_pred = np.arange(-0.11,0.11,0.001) # pred values need to have same spacing for toeplitz!
flux = MF.Transit_aRs(mfp,time) + hp[0]*np.sin(2*np.pi*10.*time) + np.random.normal(0,hp[-1],time.size)

#define the GP
gp = GeePea.GP(time,flux,p=mfp+hp,kf=GeePea.ToeplitzSqExponential,mf=MF.Transit_aRs,ep=ep,x_pred=time_pred)

#optimise the free parameters
gp.optimise()

#and plot
gp.plot()
for i in range(3): pylab.plot(gp.xmf_pred, gp.getRandomVector())