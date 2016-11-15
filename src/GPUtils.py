"""
Some non-core utility functions for GPs
"""

import numpy as np
import pylab

####################################################################################################

def RandomVector(K,m=None):
  """
  Get a random gaussian vector from the covariance matrix K.
  """
  
  if m is None: #set mean function if not given
    m = np.zeros(K[:,0].size)
  
  return np.random.multivariate_normal(m,K)

def RandVectorFromConditionedGP(K_s,PrecMatrix,K_ss,r,m=None):
  """
  Get a random gaussian vector from the covariance matrix K.
  m - mean function
  calculates conditional covariance K_ss
  calculates conditional mean and adds to mean function
  """
  
  #ensure all data are in matrix form
  K_s = np.matrix(K_s)
  K_ss = np.matrix(K_ss)
  PrecMatrix = np.matrix(PrecMatrix)
  r = np.matrix(np.array(r).flatten()).T # (n x 1) column vector

  # (q x n) = (q x n) * (n x n) * (n x 1)
  f_s = K_s * PrecMatrix * r
   
  # (q x q) = (q x q) - (q x n) * (n x n) * (n x q)  
  K_ss_cond = K_ss - np.matrix(K_s) * PrecMatrix * np.matrix(K_s).T

  if m is None: #set zero mean function if not given
    m = np.zeros(f_s.size)
  
  return RandomVector(K_ss_cond,m=np.array(f_s).flatten()+m)

def PlotRange(ax,x,y,y_err,sigma=1.0,facecolor='0.5'):
  """
  Plot a range 'area' for GP regression given x,y values, y_error and no. sigma
  """
  y1,y2 = y+sigma*y_err, y-sigma*y_err
  
  ax.fill_between(x, y1, y2, where=y1>=y2, facecolor=facecolor)

def PlotSigmas(x,y,y_err,nsig=3,color='r',alpha=0.5):
  """
  Plot 1 and 2 sigma range areas for GP regression given x,y values, y_error
  """
    
  for i in np.arange(-nsig,nsig+1):  
    pylab.plot(x,y+y_err*i,'-',color=color,alpha=alpha,lw=0.5)

def PlotDensity(x,y,yerr,n=2000,nsig=5.,cmap='gray_r',sm_x=None,supersamp=None):

  if supersamp: #resample the inputs to smooth plot
    x_new = np.linspace(x.min(),x.max(),x.size*supersamp)
    y = np.interp(x_new,x,y)
    yerr = np.interp(x_new,x,yerr)
    x = x_new
  
  #set range of y
  y_lower,y_upper = (y-nsig*yerr).min(),(y+nsig*yerr).max()
  y_range = np.linspace(y_lower,y_upper,n)
  
  XX,YY = np.meshgrid(x,y_range)

  IM = np.exp(-((YY-y)**2 / yerr**2)/2.)

  #smooth in x?
  if sm_x:
    IM = ndimage.gaussian_filter1d(IM, sigma=sm_x, axis=1)  
  #IM = ndimage.median_filter(IM, footprint=(1,3))  

  #mask the array below nsig sigma - this allows overlapping transits, and presumably
  #lowers file size
  MaskedIM = np.ma.masked_where(IM<np.exp(-0.5*nsig**2),IM)
  
  pylab.imshow(MaskedIM, cmap=cmap, aspect='auto', origin='lower', extent=(x[0], x[-1], \
    y_lower, y_upper),vmin=np.exp(-0.5*nsig**2),vmax=1,interpolation='gaussian',alpha=1.0)
  
  return IM

def PlotRanges(x,y,y_err,lc='k',ls='-',title=None,lw=2,lw2=-1,c2='0.8',c1='0.6'):
  """
  Plot 1 and 2 sigma range areas for GP regression given x,y values, y_error
  """
  
  ax = pylab.gca()

  ax.plot(x, y, color=lc, linewidth=lw, linestyle=ls) #plot predictive function and ranges
  if lw2 < 0: lw2 = lw/2.
  
  y1,y2 = y+2*y_err, y-2*y_err
  ax.fill_between(x, y1, y2, where=y1>=y2, facecolor=c2,lw=lw2)
  
  y1,y2 = y+1*y_err, y-1*y_err
  #plot thicker lines for 1sigma lims
  #ax.plot(x, y1, 'k-', linewidth=2)
  #ax.plot(x, y2, 'k-', linewidth=2)
  
  ax.fill_between(x, y1, y2, where=y1>=y2, facecolor=c1,lw=lw2)
  
  pylab.plot()
  
  if title: pylab.title(title)

def PlotData(x,y,y_err,title=None,**kwargs):
  """
  Plot 1 and 2 sigma range areas for GP regression given x,y values, y_error
  """

  ax = pylab.gca()

  ax.errorbar(x,y,yerr=y_err,fmt='k.',**kwargs)
  
  if title: pylab.title(title)
  
  pylab.plot()
  
def PlotRange3D(ax,x1_pred,x2_pred,f_pred,f_pred_err,sigma=1.,facecolor=['r','g'],plot_range=True):
  """
  Plot a range 'surface' for GP regression given X,f values, f_error and no. sigma
  onto 3D axis 'ax'
  """
  from matplotlib.mlab import griddata
  
  #create X,Y mesh grid
  xi, yi = np.arange(x1_pred.min(),x1_pred.max(),0.1), np.arange(x2_pred.min(),x2_pred.max(),0.1)
  X, Y = np.meshgrid(xi, yi)

  #use grid data to place (x1_pred, x2_pred, f_pred) values onto Z grid
  Z = griddata(x1_pred, x2_pred, f_pred, xi, yi) #grid the predicted data
  Z_u = griddata(x1_pred, x2_pred, f_pred+f_pred_err*sigma, xi, yi) #and error data...
  Z_d = griddata(x1_pred, x2_pred, f_pred-f_pred_err*sigma, xi, yi)

  #plot the surfaces on the axis (must be passed a 3D axis)
  ax.plot_wireframe(X,Y,Z,color=facecolor[0],rstride=1,cstride=1)
  if plot_range:
    ax.plot_wireframe(X,Y,Z_u,color=facecolor[1],rstride=2,cstride=2)
    ax.plot_wireframe(X,Y,Z_d,color=facecolor[1],rstride=2,cstride=2)

####################################################################################################
def add_n_par(N):
  """
  Simple decorator function to add n_par to a static function - required for built in mean function
  """
  def decor(func):
    func.n_par = N
    return func
  return decor

###############################################################################################################
