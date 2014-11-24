
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time

##########################################################################################

def MCMC_N(LogPosterior,gp,post_args,ch_len,ep,N=1,chain_filenames=['MCMC_chain_1'],\
  adapt_limits=(0,0,0),glob_limits=(0,0,0),thin=1,orth=0,acc=0.234,max_ext=0,ext_len=None,GRtarget=1.01):
  """
  edit: 6 May 2014
  MCMC_N forked from standard MCMC
  idea is to run multiple chains simultaneously, and compute the GR stat periodically until
  convergence is reached
  N is number of chains to run at once
  
  Could speed this up by parallelising the inner loop over N chains, however, the overheads
  with multiprocessing were too large. Would be better off splitting the main while loop
  into groups of range(i,var_ch_len) every time extensions are added to the chains and
  parallelising the chains that way. However most of the time I'm running multiple light
  curve fits so easier (and quicker) just to run separate scripts!
  
  Python adaption of MyMCMC (C) code. Marginally slower than the C version, but slowdown
  is insignificant when using an expensive posterior function, ie for GPs. Should be much
  easier to adapt and use better adaption algorithms to make up for this, although need to
  be very careful with choice of python functions, as these have significant effects on
  the efficiency.
  
  Note it is very inefficient to have expressions within the chain loop, so as much as
  possible, calculations should be done outside the loop in large arrays, eg the random
  numbers are generated for the whole chain at once (and multiplied by stepsizes). This
  is also done when the stepsizes are adapted, which is much much quicker than doing it
  at each step of the loop.
  
  LogPosterior - log posterior distribution
  gp - array/list of guess parameters
  ep - array/list of (initial) steps
  chain_filenames - list of chain filenames, used to calculate no of chains
  n_chains - no of chains, overwrites chains_filenames, uses generic names for chains
  adapt_limits/glob_limits - tuple of (lower,upper,number), defining the range and number
    of adaptions within that range. adapt for relative step sizes and glob for the global
    stepsize
  thin - int >= 1 - thin the chains by this amount, ie only output every 'thin' steps of
    the chain
  orth - default is to use the covariance matrix at each adaption step and make correlated
    steps in parameter space, this flag forces orthogonal steps, ie along each axis
    independently
  acc - target acceptance ratio - for infinite iid Gaussian dist -> 23.4%, for single par
    is 44%
  
  """
  
  #first set chain filenames
  if N < 1: print "N must be at least 1!"; return
  if N > 1: chain_filenames = ["MCMC_chain_%d" % ch_no for ch_no in range(1,N+1)]
  N = len(chain_filenames) # if chain_filenames are set instead of N, but N>1 overwrites with prev line
  
  #set chain len and extension len of chain, default to chain length
  if ext_len == None: ext_len = ch_len
  var_ch_len = ch_len #start with double the original chain length
  
  #set burn in value, only used for the GR calculation to check convergence
  burn_in = np.max([adapt_limits[1],glob_limits[1]])
  if burn_in == 0: burn_in = ch_len/2
  
  #print parameters at start of chain
  PrintParams(chain_filenames,var_ch_len,burn_in,ext_len,max_ext,LogPosterior,adapt_limits,glob_limits,gp,ep)
  print '-' * 80
  
  #initialise parameters
  p,e = np.array([gp,]*N),np.array([ep,]*N)
  p_acc,L_acc = np.copy(p),np.array([-np.inf,]*N)
  
  #arrays for storing results
  ParArr = np.zeros((N,var_ch_len/thin,len(p[0])))
  PostArr = np.zeros((N,var_ch_len/thin))
  AccArr = np.zeros((N,var_ch_len))
  
  #jump parameters, error array computed in advance - much faster to compute as a block
  G = np.array([np.float(2.4**2/(e[0]>0).sum()),]*N)
  K = np.array([np.diag(e[0]**2),]*N)
  np.random.seed()
  RandArr = np.random.multivariate_normal(np.zeros(p[0].size),K[0],(N,var_ch_len)) * G[0]
  RandNoArr = np.random.rand(N,var_ch_len)
    
  ####### loop over chain iterations ###############
  start = time.time()
  i = 0
  while i<var_ch_len: #use while loop as var_ch_len is adjusted
    if i % ((var_ch_len)/20) == 0:
      PrintBar(i,chain_filenames,ch_len,var_ch_len,AccArr,start)  

    ################ MH step for each chain ################
    #should look into parallelising this step - could speed up significantly
    #tried but seems overheads are not really worth it esp when running multiple light curves!
    for n in range(N): #easy to parallelise n here!
      p_prop = p_acc[n] + RandArr[n][i]
      L_prop = LogPosterior(p_prop,*post_args)
      #Metropolis algorithm to accept step
#      if np.random.rand() < np.exp(L_prop - L_acc[n]):
      if RandNoArr[n][i] < np.exp(L_prop - L_acc[n]):
        p_acc[n],L_acc[n] = p_prop,L_prop
        AccArr[n][i] = 1 #update acceptance array
      #add new posterior and parameters to chain
      if i%thin==0: ParArr[n][i/thin],PostArr[n][i/thin] = p_acc[n],L_acc[n]

    ################ Adaptive step sizes ################
    #adaptive stepsizes - shouldn't be executed after arrays extended in current form!
    if (i <= adapt_limits[1]) and (i > adapt_limits[0]):
      if (i-adapt_limits[0]) % ((adapt_limits[1]-adapt_limits[0])/adapt_limits[2]) == 0:
        for n in range(N):
          if orth: K[n] = np.diag(((e + 4*ParArr[n][adapt_limits[0]/thin:i/thin].std(axis=0))/5.)**2.) #for diagonal covariance matrix
          else: K[n] = (K[n] + 4.*np.cov(ParArr[n][adapt_limits[0]/thin:i/thin],rowvar=0))/5.
          K[n][np.where(e[n]==0.)],K[n][:,np.where(e[n]==0.)] = 0.,0. #reset error=0. values to 0.
          RandArr[n,i:] = np.random.multivariate_normal(np.zeros(p[n].size),K[n],var_ch_len-i) * G[n]
          RandArr[n,i:][:,np.where(e[n]==0.)[0]] = 0. #set columns to zero after too!
    #adaptive global step size
    if (i <= glob_limits[1]) and (i > glob_limits[0]):
      if (i-glob_limits[0]) % ((glob_limits[1]-glob_limits[0])/glob_limits[2]) == 0:
        for n in range(N):
          G[n] *= (1./acc) *  min(0.9,max(0.1,AccArr[n][i-(glob_limits[1]-glob_limits[0])/glob_limits[2]:i].sum()/((glob_limits[1]-glob_limits[0])/glob_limits[2])))
          RandArr[n,i:] = np.random.np.random.multivariate_normal(np.zeros(p[n].size),K[n],var_ch_len-i) * G[n]
          RandArr[n,i:][:,np.where(e[n]==0.)[0]] = 0.

    ################ Test Convergence ################    
    #test convergence and extend chains if needed, unless max extensions has been reached
    if i+1 >= ch_len and (i-ch_len+1)%ext_len == 0 and N > 1:
      grs = ComputeGRs(np.array(ep),ParArr[:,burn_in:,:])
      if grs.max() > GRtarget:
        if max_ext > 0:
          max_ext -= 1 #decrement the max extensions int
          PrintBar(i,chain_filenames,ch_len,var_ch_len,AccArr,start,extra_st=" extending chain ({}, GR max = {:.4f}) ".format(var_ch_len+ext_len,grs.max()))
          RandArr = np.concatenate([RandArr,np.random.multivariate_normal(np.zeros(p[0].size),K[0],(N,ext_len)) * G[0]],axis=1)
          RandArr[:,i:,np.where(e[n]==0.)[0]] = 0. #set columns to zero after too!
          RandNoArr = np.concatenate([RandNoArr,np.random.rand(N,ext_len)],axis=1)
          ParArr = np.concatenate([ParArr,np.zeros((N,ext_len/thin,len(p[0])))],axis=1)
          PostArr = np.concatenate([PostArr,np.zeros((N,ext_len/thin))],axis=1)
          AccArr = np.concatenate([AccArr,np.zeros((N,ext_len))],axis=1)   
          var_ch_len += ext_len
        else:
          PrintBar(i,chain_filenames,ch_len,var_ch_len,AccArr,start,extra_st=" not converged (GR max = {:.4f})                ".format(grs.max()))
          grs = ComputeGRs(np.array(ep),ParArr[:,burn_in:,:])
      else:
        PrintBar(i,chain_filenames,ch_len,var_ch_len,AccArr,start,extra_st=" converged (GR max = {:.4f})                 ".format(grs.max()))

    #################################################
    
    #increment i
    i += 1
    
    ####### end loop over chains ###########

  PrintBar(i,chain_filenames,ch_len,var_ch_len,AccArr,start,True,"",end_st="Final chain len = {}".format(i))
  
  for n,chain in enumerate(chain_filenames):
    
    np.save(chain+".npy",np.concatenate([PostArr[n].reshape(PostArr[n].size,1),ParArr[n]],axis=1))
  
  ####### end loop over chains ############
  print '-' * 80
  
##########################################################################################

def PrintBar(i,ch_files,ch_len,var_ch_len,AccArr,start,finish=False,extra_st="",end_st=""):
  ts = time.time()-start
  N = len(ch_files)
  print "Running {} chains:".format(N) + extra_st
  for n in range(N):
    a_str = "" if i <= ch_len/5 else ", acc = %.2f%%  " % (100.*np.float(AccArr[n][ch_len/5:i].sum())/(i-ch_len/5))
    print u" chain %s: '%s' \033[31m%-21s\033[0m t = %dm %.2fs%s" % (n+1,ch_files[n],'#'*(i/(var_ch_len/20)+1),ts // 60., ts % 60.,a_str)
  sys.stdout.write('\033[{}A'.format(N+1))
  if finish: print "\n"*(N+1) + end_st

##########################################################################################

def PrintParams(ch_filenames,ch_len,burn_in,ext_len,max_ext,posterior,adapt_limits,glob_limits,gp,ep):

  print "Simultaneous MCMC chains runnning..."
  print " No Chains: %d" % len(ch_filenames)
  print " Chain Length: %d" % ch_len
  print " Burn in: %d" % burn_in
  print " Max {} extensions of length {}".format(max_ext,ext_len)
  if(adapt_limits[2]): print " Relative-step adaption limits: (%d,%d,%d)" % (adapt_limits[0],adapt_limits[1],adapt_limits[2])
  if(glob_limits[2]): print " Global-step adaption limits: (%d,%d,%d)" % (glob_limits[0],glob_limits[1],glob_limits[2])
  print " Computing {} chains simultaneously: {}".format(len(ch_filenames),ch_filenames)
  print " Posterior probability function: ", posterior
  print " Function params <value prop_size>:"
  for q in range(len(gp)):
    print "  p[%d] = %f +- %f" % (q,gp[q],ep[q])

##########################################################################################

def ComputeGRs(ep,ParArr,conv=0):
  """
  Compute the Gelman and Rubin statistic and errors for all variable parameters
  """
  
  p = np.where(np.array(ep)>0)[0]
  GRs = np.zeros(len(p))
  
  for q,i in enumerate(p):
    
    #get mean and variance for the two chains
    mean = ParArr[:,:,i].mean(axis=1)
    var = ParArr[:,:,i].var(axis=1)
    
    #get length of chain
    L = len(ParArr[0,:,0])
    
    #and calculate the GR stat
    W = var.mean(dtype=np.float64) #mean of the variances
    B = mean.var(dtype=np.float64) #variance of the means
    GR = np.sqrt((((L-1.)/L)*W + B) / W) #GR stat
    
    GRs[q] = GR
  
  return GRs

##########################################################################################

  
  