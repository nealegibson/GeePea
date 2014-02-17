
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time

##########################################################################################

def MCMC(LogPosterior,gp,post_args,ch_len,ep,chain_filenames=['MCMC_chain'],n_chains=0,\
  adapt_limits=(0,0,0),glob_limits=(0,0,0),thin=1,orth=0,acc=0.234):
  """
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
  if n_chains > 0: chain_filenames = ["MCMC_chain_%d" % ch_no for ch_no in range(1,n_chains+1)]
  
  #print parameters
  PrintParams(chain_filenames,ch_len,LogPosterior,adapt_limits,glob_limits,gp,ep)
  print '-' * 80
  
  ####### loop over chains ###############
  for n,chain in enumerate(chain_filenames):

    #initialise parameters
    p,e = np.copy(gp),np.copy(ep)
    p_acc,L_acc = np.copy(p),-np.inf    
    #arrays for storing results
    ParArr = np.zeros(ch_len/thin*len(p)).reshape(ch_len/thin,len(p))
    PostArr = np.zeros(ch_len/thin)
    AccArr = np.zeros(ch_len)

    #jump parameters
    #error array computed in advance - much faster to compute as a block
    G = np.float(2.4**2 / (e>0).sum() )
    K = np.diag(e**2) #create starting (diagonal) covariance matrix
    #RA = np.random.normal(0.,1.,len(p)*ch_len).reshape(ch_len,len(p)) * e * G
    np.random.seed()
    RandArr = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len) * G
    
    #print "Computing Chain %d: '%s' " % (n+1,chain),
    ####### individual chain ###############
    start = time.time() 
    for i in xrange(ch_len):
      if i % ((ch_len)/20) == 0:
        PrintBar(n,chain,i,ch_len,AccArr,start)
      
      #create proposal parameters, and calculate posterior
#      p_prop = p_acc + RandArr[i] * e * G #for diag covariance matrix
      p_prop = p_acc + RandArr[i] #this line has the largest (extra) overhead from C -> python version
      L_prop = LogPosterior(p_prop,*post_args)

      #Metropolis algorithm to accept step
      if np.random.rand() < np.exp(L_prop - L_acc):
        p_acc,L_acc = p_prop,L_prop
        AccArr[i] = 1 #update acceptance array

      #add new posterior and parameters to chain
      if i%thin==0: ParArr[i/thin],PostArr[i/thin] = p_acc,L_acc
      
      #adaptive stepsizes
      if (i <= adapt_limits[1]) and (i > adapt_limits[0]):
        if (i-adapt_limits[0]) % ((adapt_limits[1]-adapt_limits[0])/adapt_limits[2]) == 0:
          #RA = np.random.normal(0.,1.,len(p)*ch_len).reshape(ch_len,len(p)) * e * G
          if orth: K = np.diag(((e + 4*ParArr[adapt_limits[0]/thin:i/thin].std(axis=0))/5.)**2.) #for diagonal covariance matrix
          else: K = (K + 4.*np.cov(ParArr[adapt_limits[0]/thin:i/thin],rowvar=0))/5.
          K[np.where(e==0.)],K[:,np.where(e==0.)] = 0.,0. #reset error=0. values to 0.
          RandArr[i:] = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len-i) * G
          RandArr[i:][:,np.where(e==0.)[0]] = 0. #set columns to zero after too!
      #adaptive global step size
      if (i <= glob_limits[1]) and (i > glob_limits[0]):
        if (i-glob_limits[0]) % ((glob_limits[1]-glob_limits[0])/glob_limits[2]) == 0:
          G *= (1./acc) *  min(0.9,max(0.1,AccArr[i-(glob_limits[1]-glob_limits[0])/glob_limits[2]:i].sum()/((glob_limits[1]-glob_limits[0])/glob_limits[2])))
          RandArr[i:] = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len-i) * G
          RandArr[i:][:,np.where(e==0.)[0]] = 0.
          
    ####### end individual chain ###########
    PrintBar(n,chain,i,ch_len,AccArr,start); print
    np.save(chain+".npy",np.concatenate([PostArr.reshape(PostArr.size,1),ParArr],axis=1))

  ####### end loop over chains ############
  print '-' * 80
  
##########################################################################################

def PrintBar(n,chain,i,ch_len,AccArr,start):
  "Print the status bar - probably a more elegant way to write this..."
  ts = time.time()-start
  a_str = "" if i <= ch_len/5 else ", acc = %.2f%%" % (100.*np.float(AccArr[ch_len/5:i].sum())/(i-ch_len/5+1))
  print "\rComputing Chain %d: '%s' %-20s t = %dm %.2fs%s" % (n+1,chain,'#'*(i/(ch_len/20)+1),ts // 60., ts % 60.,a_str),
  sys.stdout.flush();

##########################################################################################

def PrintParams(ch_filenames,ch_len,posterior,adapt_limits,glob_limits,gp,ep):

  print "Infer.MCMC runnning..."
  print "MCMC parameters:"
  print " No Chains: %d" % len(ch_filenames)
  print " Chain Length: %d" % ch_len
  if(adapt_limits[2]): print " Relative-step adaption limits: (%d,%d,%d)" % (adapt_limits[0],adapt_limits[1],adapt_limits[2])
  if(glob_limits[2]): print " Global-step adaption limits: (%d,%d,%d)" % (glob_limits[0],glob_limits[1],glob_limits[2])
  print " Computing chains:", ch_filenames
  print " Posterior probability function: ", posterior
  print " Function params <value prop_size>:"
  for q in range(len(gp)):
    print "  p[%d] = %f +- %f" % (q,gp[q],ep[q])

##########################################################################################

  
  