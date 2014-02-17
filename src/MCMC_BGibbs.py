  
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time

##########################################################################################

def BGMCMC(LogPosterior,gp,post_args,ch_len,ep,gibbs_index,chain_filenames='MCMC_chain',n_chains=0,\
  adapt_limits=(0,0,0),glob_limits=(0,0,0),thin=1,orth=0,acc=0.234):
  """
  Generalisation of the MCMC.py code to allow for blocked Gibbs sampling. See MCMC
  docstring for details. Here I've added the addition of a Gibbs index, which is just an
  array of the same size as the inputs, which includes the order in which to index. 0
  indicates the parmeters is not to vary.
  
  eg gibbs_index = [0,1,0,1,2,2,3] will vary parameters 1, 2 then 3 in turn, evaluating
  the log posterior at each and accepting them according to the MH rule.
  
  Note the logPosterior will be evaluated ch_len * no_gibbs_steps times, and is especially 
    useful when using the InferGP class posterior, which only constructs/inverts the covariance
    when necessary, and stores previous results
  
  acc - target acceptance ratio - for infinite iid Gaussian dist -> 23.4%, for single par
    is 44%. These will be varied independently for each block, and can have a different target
    acceptance if an array is provided
  
  When orth=0 (ie default) be aware that parameters in different blocks may still be
    correlated. This is taken into account in the separate scaling to some degree, but
    highly correlated variables should probably be in the same block, or alternatively set
    orth = 1 for orthogonal steps (an intermediate solution is possible, but I couldn't be
    bothered coding it right now)
  
  """
  
  #first set chain filenames
  if n_chains > 0: chain_filenames = ["MCMC_chain_%d" % ch_no for ch_no in range(1,n_chains+1)]
  
  #print parameters
  PrintParams(chain_filenames,ch_len,LogPosterior,adapt_limits,glob_limits,gp,ep,gibbs_index)
  print '-' * 80
  
  #prep gibbs array
  gibbs = np.array(gibbs_index)
  no_steps = gibbs.max()
  gi = range(no_steps)
  for q in range(1,no_steps+1):
    gi[q-1] = np.where(gibbs==q) #get index to the gibbs steps

  ####### loop over chains ###############
  for n,chain in enumerate(chain_filenames):

    #initialise parameters
    p,e = np.copy(gp),np.copy(ep)
    p_acc,L_acc = np.copy(p),-np.inf    
    #arrays for storing results
    ParArr = np.zeros(ch_len/thin*len(p)).reshape(ch_len/thin,len(p))
    PostArr = np.zeros(ch_len/thin)
    AccArr = np.zeros(ch_len*no_steps).reshape(ch_len,no_steps) #acceptance rate for each Gibbs block
    
    #jump parameters
    #error array computed in advance - much faster to compute as a block
    G = np.zeros(no_steps) #set default G depending on no of varying parameters per block
    for q in range(1,no_steps+1): G[q-1] = (2.4**2/(gibbs==q).sum())
    Garr = np.array([G[v-1] if v>0 else 0 for v in gibbs])
    
    ACC = np.ones(no_steps) * acc
    
    K = np.diag(e**2) #create starting (diagonal) covariance matrix
    #RA = np.random.normal(0.,1.,len(p)*ch_len).reshape(ch_len,len(p)) * e * G
    np.random.seed()
    RandArr = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len) * Garr
    
    #print "Computing Chain %d: '%s' " % (n+1,chain),
    start = time.time() 
    ####### individual chain ###############
    for i in xrange(ch_len):
      if i % ((ch_len)/20) == 0:
        PrintBar(n,chain,i,ch_len,AccArr,start,no_steps)
        #sys.stdout.write('#'); sys.stdout.flush();
      
      #Blocked Gibbs algorithm
      #cycle over Gibbs steps
      for q in range(no_steps):
        #gi = np.where(gibbs==q) #get index to the gibbs steps
        #print "step = ",q,
        p_prop = np.copy(p_acc)
        p_prop[gi[q]] += RandArr[i][gi[q]]
        #print p_prop
        L_prop = LogPosterior(p_prop,*post_args)
        #Metropolis algorithm to accept step
        if np.random.rand() < np.exp(L_prop - L_acc):
          p_acc,L_acc = p_prop,L_prop        
          AccArr[i][q] = 1 #update acceptance array (store no. acceptances for gibbs)
        #  print "acc"
          
      #add new posterior and parameters to chain
      if i%thin==0: ParArr[i/thin],PostArr[i/thin] = p_acc,L_acc
      
      #adaptive stepsizes
      if (i <= adapt_limits[1]) and (i > adapt_limits[0]):
        if (i-adapt_limits[0]) % ((adapt_limits[1]-adapt_limits[0])/adapt_limits[2]) == 0:
          #RA = np.random.normal(0.,1.,len(p)*ch_len).reshape(ch_len,len(p)) * e * G
          if orth: K = np.diag(((e + 4*ParArr[adapt_limits[0]/thin:i/thin].std(axis=0))/5.)**2.) #for diagonal covariance matrix
          else: K = (K + 4.*np.cov(ParArr[adapt_limits[0]/thin:i/thin],rowvar=0))/5.
          K[np.where(e==0.)],K[:,np.where(e==0.)] = 0.,0. #reset error=0. values to 0.
          RandArr[i:] = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len-i) * Garr
          RandArr[i:][:,np.where(e==0.)[0]] = 0. #set columns to zero after too!
      #adaptive global step size
      if (i <= glob_limits[1]) and (i > glob_limits[0]):
        if (i-glob_limits[0]) % ((glob_limits[1]-glob_limits[0])/glob_limits[2]) == 0:
          for q in range(no_steps): #update G for each block
            G[q] *= (1./ACC[q]) *  min(0.9,max(0.1,AccArr[:,q][i-(glob_limits[1]-glob_limits[0])/glob_limits[2]:i].sum()/((glob_limits[1]-glob_limits[0])/glob_limits[2])))
          Garr = np.array([G[v-1] if v>0 else 0 for v in gibbs])
          RandArr[i:] = np.random.np.random.multivariate_normal(np.zeros(p.size),K,ch_len-i) * Garr
          RandArr[i:][:,np.where(e==0.)[0]] = 0.
          #print G
          
    ####### end individual chain ###########
    PrintBar(n,chain,i,ch_len,AccArr,start,no_steps); print
    np.save(chain+".npy",np.concatenate([PostArr.reshape(PostArr.size,1),ParArr],axis=1))

  ####### end loop over chains ############
  print '-' * 80
  
##########################################################################################

def PrintBar(n,chain,i,ch_len,AccArr,start,no_steps):
  ts = time.time()-start
  if i <= ch_len/5:
    a_str = ""
    a_str2 = ""
  else:
    a_str = "" if i <= ch_len/5 else ", acc = %.2f%%" % (100.*np.float(AccArr[ch_len/5:i].sum())/no_steps/(i-ch_len/5+1))
    a_str2 = "["+"".join(["%.2f%%," % (100.*np.float(AccArr[ch_len/5:i].sum(axis=0)[q])/(i-ch_len/5+1)) for q in range(no_steps)])+"\b]"
  print "\rComputing Chain %d: '%s' %-20s t = %dm %.2fs%s" % (n+1,chain,'#'*(i/(ch_len/20)+1),ts // 60., ts % 60.,a_str),
  print a_str2,
  sys.stdout.flush();

##########################################################################################
  
def PrintParams(ch_filenames,ch_len,posterior,adapt_limits,glob_limits,gp,ep,gibbs):

  print "Infer.BGMCMC runnning..."
  print "Blocked Gibbs MCMC parameters:"
  print " No Chains: %d" % len(ch_filenames)
  print " Chain Length: %d" % ch_len
  if(adapt_limits[2]): print " Relative-step adaption limits: (%d,%d,%d)" % (adapt_limits[0],adapt_limits[1],adapt_limits[2])
  if(glob_limits[2]): print " Global-step adaption limits: (%d,%d,%d)" % (glob_limits[0],glob_limits[1],glob_limits[2])
  print " Computing chains:", ch_filenames
  print " Posterior probability function: ", posterior
  print " Function params <value prop_size [block]>:"
  for q in range(len(gp)):
    print "  p[%d] = %f +- %f [%d]" % (q,gp[q],ep[q],gibbs[q])

##########################################################################################

  
  