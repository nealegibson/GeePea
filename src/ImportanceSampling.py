
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time
import pylab

##########################################################################################

def ImportanceSamp(Posterior,post_args,m,K,no_samples,filename="ImSamp"):
  """
  Importance sampler using a Gaussian proposal distribution. See the PRML book chapter 11
  for a description of the method. Not as efficient as MCMC, but can be a useful check on
  the results.
  
  m,K - mean and covariance matrix of proposal dist (e.g. est from MCMC)
  Posterior - Posterior function with parameter array as first argument
  post_args - tuple of remaining (fixed) args to Posterior function
  no_samples - no of samples to take from the proposal distribution
  filename - stem of output file to store weights filename.npy 
  
  """

  print "Importance Sampler runnning..."
  print "Gaussian proposal"
  print " No Samples: %d" % no_samples
  
  #first compress matrix - ie remove non-variable columns, or we have a singular matrix
  no_pars = np.diag(K).size
  var_par = np.diag(K)>0
  Ks = K.compress(var_par,axis=0)
  Ks = Ks.compress(var_par,axis=1)
  ms = m[var_par]
  
  #calculate the inverse covariance matrix and log determinant (of compressed matrix)
  invKs = np.mat(Ks).I
  sign,logdetKs = np.linalg.slogdet(Ks) #calcualte detK
  
  #create array of parameters to sample from
  SampArr = np.random.np.random.multivariate_normal(ms,Ks,no_samples)
  #and create full array (ie with constant values too)
  FullSampArr = np.empty(no_samples*no_pars).reshape(no_samples,no_pars)
  FullSampArr[:] = m
  FullSampArr[:,var_par] = SampArr
  
  #create storage arrays for probabilities
  PostProb = np.empty(no_samples)
  PropProb = np.empty(no_samples)
  
  #calculate difference between maximum likelihoods (used to ease numerical calculations)
  Z_diff = Posterior(m,*post_args) - NormalDist(ms,invKs,logdetKs,ms)
  print " log Bayes evidence from Gauss =", Z_diff
  
  #loop over samples and calculate the probabilities
  start = time.time() 
  print '-' * 80
  print "Sampling proposal distribution ",
  for i in xrange(no_samples):
    if i % ((no_samples)/20) == 0: sys.stdout.write('.'); sys.stdout.flush();
    #calculate and store posterior (subtract Z_diff from logPost - same as dividing posterior by a constant)
    PostProb[i] = Posterior(FullSampArr[i],*post_args) - Z_diff
    #calculate and store proposal probability
    PropProb[i] = NormalDist(ms,invKs,logdetKs,SampArr[i])
    #print PostProb[i],PropProb[i]
  print " t = %.2f s" % (time.time()-start)
  print '-' * 80

  #Calculate importance weights
  Weights = np.exp(PostProb - PropProb)
  
  #need to calculate normalised weights, as the posterior is not normalised (and we need to work out Z anyway)
#   Z_ratio = np.mean(Weights) # the mean difference in scale between the distributions
#   w = Weights / Weights.sum() # normalise the weights to work out expectations - ie must sum to 1 in expectation eq
  
#   #print out the mean, std per param and evidence
#   print "Sampled distribution:"
    # Bayes evidence - Z_ratio gives the scale difference between posterior and prior (normalised)
    # Then add on Z_diff to log - to account for arbitrary scaling from earlier
#   print " log Bayes evidence =", np.log(Z_ratio) + Z_diff
#   print " par = mean +- gauss_err"
#   mean,var = np.zeros(no_pars),np.zeros(no_pars)
#   for i in range(no_pars): #loop over parameters and get parameters, errors, and GR statistic
#     mean[i] = np.sum(w*FullSampArr[:,i]) # calculate expectation value of x using normalised weights
#     var[i] = np.sum(w*(FullSampArr[:,i]-mean[i])**2)  # and 2nd central moment/variance
#     print " p[%d] = %.7f +- %.7f" % (i,mean[i],np.sqrt(var[i]))
  
  #save data file
  np.save(filename+".npy",np.concatenate([Weights.reshape(Weights.size,1),FullSampArr],axis=1))
  
  #Analyse the Importance samples
  return AnalyseImportanceSamp(Z_diff,filename)
    
##########################################################################################

def AnalyseImportanceSamp(Z_diff=0,filename="ImSamp",filter=None):

  Data = np.load(filename+".npy")
  FullSampArr = Data[:,1:] #take slice of data for fullsamp array
  
  no_pars = Data[0].size-1
  
  if filter==None:
    index_list = [np.arange(Data[:,0].size)]
  else:
    index_list = [np.arange(q,Data[:,0].size,filter) for q in range(filter)]

  for index in index_list:

    Weights = Data[:,0][index]
    Z_ratio = np.mean(Weights)
    w = Weights / Weights.sum()

    #print out the mean, std per param and evidence
    print ""
    print "Sampled distribution:"
    print " log Bayes evidence =", np.log(Z_ratio) + Z_diff
    if Z_diff == 0: print "warning no Z_diff provided, evidence is probably wrong!"
    print " par = mean +- gauss_err"
    mean,var = np.zeros(no_pars),np.zeros(no_pars)
    for i in range(no_pars): #loop over parameters and get parameters, errors, and GR statistic
      mean[i] = np.sum(w*FullSampArr[:,i][index])
      var[i] = np.sum(w*(FullSampArr[:,i][index]-mean[i])**2)
      print " p[%d] = %.7f +- %.7f" % (i,mean[i],np.sqrt(var[i]))

  if filter==None: return np.log(Z_ratio) + Z_diff, mean, np.sqrt(var)
 
##########################################################################################

def BayesFactor(logE1, logE2, H1='H1', H2='H2'):
  
  Bayes_factor = np.exp(logE1-logE2)
  dB = 10*np.log(Bayes_factor if Bayes_factor>1. else 1./Bayes_factor)

  print "Bayes factor O(%s/%s) = " % (H1,H2), Bayes_factor
  print " %s favoured by dB = %f" % (H1 if Bayes_factor>1. else H2, dB)
  
  return Bayes_factor, dB

##########################################################################################

def NormalFromMCMC(conv_length,n_chains=None,chain_filenames=None,plot=False):
  """
  Need a simple function to get the covariance matrix from a set of MCMC chains...

  """

  #get chain names tuple
  if n_chains == None and chain_filenames == None:
    chain_filenames=("MCMC_chain",)  
  if n_chains != None:
    chain_filenames = []
    for i in range(n_chains):
      chain_filenames.append("MCMC_chain_%d" % (i+1))
  
  #merge files into large matrix
  X = np.load(chain_filenames[0]+'.npy')[conv_length:] #read in data file  
  for i in range(1,len(chain_filenames)):
    X = np.concatenate((X,np.load(chain_filenames[i]+'.npy')[conv_length:]))
  
  m = X[:,1:].mean(axis=0)
  K = np.cov(X[:,1:]-m,rowvar=0) #better to mean subtract first to avoid rounding errors
  
  print "Gaussian approx:"
  print " par = mean +- gauss_err"
  for i in range(m.size): #loop over parameters and get parameters, errors, and GR statistic
    print " p[%d] = %.7f +- %.7f" % (i,m[i],np.sqrt(K[i,i]))

  if plot:
    p=np.where(np.diag(K)>0)[0]
    no_pars = p.size
    labels = ['p[%d]' % q for q in p]
  
    #make plot to slow the eigenvectors?
    pylab.clf() #clear figure
    pylab.rcParams['lines.markersize'] = 2.
    pylab.rcParams['font.size'] = 10
    pylab.subplots_adjust(left=0.07,bottom=0.07,right=0.93,top=0.93,wspace=0.00001,hspace=0.00001)
    
    Data = X
    index = np.random.randint(conv_length,Data[:,0].size,500) #randomly sample the data for plots
    
    for i in range(no_pars): #loop over the parameter indexes supplied
      for q in range(i+1):
        pylab.subplot(no_pars,no_pars,i*no_pars+q+1,xticks=[],yticks=[]) #select subplot
        
        if(i==q):
          hdata = pylab.hist(Data[:,p[i]+1][conv_length:],20,histtype='step',normed=1)   
          pylab.xlim(hdata[1].min(),hdata[1].max())
          pylab.ylim(hdata[0].min(),hdata[0].max()*1.1)
        else:
          pylab.plot(Data[:,p[q]+1][index],Data[:,p[i]+1][index],'.')
          #add arrows eigenvectors of (root) covariance matrix
          Ks = np.mat([[K[p[q],p[q]],K[p[q],p[i]]],[K[p[i],p[q]],K[p[i],p[i]]]])
          eval,evec = np.linalg.eig(np.array(Ks))
          vec1 = (evec[:,0]*np.sqrt(eval[0])).flatten()
          pylab.plot( [m[p[q]],m[p[q]]+vec1[0]] , [m[p[i]],m[p[i]]+vec1[1]] ,'r-', lw=2)          
          vec2 = (evec[:,1]*np.sqrt(eval[1])).flatten()
          pylab.plot( [m[p[q]],m[p[q]]+vec2[0]] , [m[p[i]],m[p[i]]+vec2[1]] ,'r-', lw=2)          
      
        if q == 0: pylab.ylabel(labels[i])
        if i == (no_pars-1): pylab.xlabel(labels[q])
  
  return m,K

##########################################################################################

def NormalDist(m,invK,logdetK,p):
  """
  Simple function to calculate the prob of a draw from a multivariate normal
  """
  
  #mean subtract the vector
  r = np.matrix(np.array(p-m).flatten()).T
  
  #and calculate the log prob
  logP = -0.5 * r.T * invK * r - 0.5 * logdetK - (r.size/2.) * np.log(2*np.pi)
  
  return np.float(logP)
  
##########################################################################################

# def GenericChainNames(n):
# 
#   filenames = []
#   for ch_no in range(1,n+1): filenames.append("MCMC_chain_%d" % ch_no)
#   return filenames
#   
# def PrintParams(ch_filenames,ch_len,posterior,adapt_limits,glob_limits,gp,ep):
# 
#   print "MyPyMCMC runnning..."
#   print "MCMC parameters:"
#   print " No Chains: %d" % len(ch_filenames)
#   print " Chain Length: %d" % ch_len
#   if(adapt_limits[2]): print " Relative-step adaption limits: (%d,%d,%d)" % (adapt_limits[0],adapt_limits[1],adapt_limits[2])
#   if(glob_limits[2]): print " Global-step adaption limits: (%d,%d,%d)" % (glob_limits[0],glob_limits[1],glob_limits[2])
#   print " Computing chains:", ch_filenames
#   print " Posterior probability function: ", posterior
#   print " Function params <value prop_size>:"
#   for q in range(len(gp)):
#     print "  p[%d] = %f +- %f" % (q,gp[q],ep[q])

# array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
# func_lib = npct.load_library("PyMCMC_functions", ".")
# func_lib.Perturb.restype = None
# func_lib.Perturb.argtypes = [array_1d_double, array_1d_double, c_double]
# def Perturb(p,e,G):
#   func_lib.Perturb(p,e,p.size,G,p_new)
##########################################################################################

  
  