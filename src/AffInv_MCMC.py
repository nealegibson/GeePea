
import numpy as np
np.seterr(divide='ignore') #ignore errors in log division
np.seterr(all='ignore') #ignore errors in log division
import sys
import time

##########################################################################################

def AffInvMCMC(LogPosterior,gp,post_args,n,ch_len,ep,chain_filenames=['MCMC_chain'],n_chains=0):
  """
  Affine invariant MCMC. ep only used for starting distribution, need n is number of walkers
  and ch_len is the number of iterations over that
    
  """
  
  #first set chain filenames
  if n_chains > 0: chain_filenames = ["MCMC_chain_%d" % ch_no for ch_no in range(1,n_chains+1)]
  
  #print parameters
  PrintParams(chain_filenames,n,ch_len,LogPosterior,gp,ep)
  print '-' * 80
  
  ####### loop over chains ###############
  for n_ch,chain in enumerate(chain_filenames):

    #initialise parameters
    p,e = np.copy(gp),np.copy(ep)
    
    a = 2.
    no_pars = e.size
    D = (e>0).sum()
    
    #arrays for storing results
    ParArr = np.zeros(ch_len*n*no_pars).reshape(ch_len*n,no_pars)
    PostArr = np.zeros(ch_len*n)
    AccArr = np.zeros(ch_len*n)

    #starting array, n_walkers x n_pars - sample from Gaussian distribution
    p_arr = np.random.normal(0,1,(n,no_pars)) * e + p
    
    ####### individual chain ###############
    #np.random.seed()
    start = time.time() 
    PrintBar(n_ch,chain,0,ch_len*n,start,AccArr)
    
    #first calculate posterior for starting array
    L_acc = np.zeros(n)
    L_prop = np.zeros(n)
    for q in range(n): #calculate the log posteriors for all n
      L_acc[q] = LogPosterior(p_arr[q],*post_args)
        
    #store first step of the chain
    ParArr[0:n],PostArr[0:n] = p_arr,L_acc
    
    #loop over chain length
    for i in xrange(1,ch_len): #start from 1
      if i % ((ch_len)/20) == 0:
        PrintBar(n_ch,chain,(i+1)*n,ch_len*n,start,AccArr)
      
      z = get_z(n,a) # get the array of z's
      rand_draw = np.random.rand(n)
      
      # for each q, need to pick a random int from 0 to n-1, but not = q
      rand_in = np.random.randint(0,n-1,n) # let the value of the index be = n as a rand draw
      rand_in[np.where(rand_in == np.arange(n))] = n-1 # ie cannot be the same as n
      
      #loop over each walker - must be updated in series - ie with the updated p_arr
      for q in range(n):
        p_prop = p_arr[rand_in[q]] + z[q] * (p_arr[q]-p_arr[rand_in[q]])
        #calculate log post for proposal step
        L_prop = LogPosterior(p_prop,*post_args)
        #accept or reject step?
#        print z[q]**np.float(D-1.), L_prop, L_acc[q]
        if rand_draw[q] < (z[q]**np.float(D-1.) * np.exp(L_prop - L_acc[q])):
#          print "Acc! Yeay!"
          AccArr[i*n+q] = 1
          p_arr[q],L_acc[q] = p_prop,L_prop
#        else:
#          print "Boo!"
      ParArr[i*n:(i+1)*n],PostArr[i*n:(i+1)*n] = p_arr,L_acc

# old version updated using the starting ensemble at each i step, doesn't preserve detailed balance
# possibly make it parallel? can divide walkers into even chunks and take the rand_in from another chunk
#       #create the trail parameter array - each one is perturbed using another random, accepted state
#       pt_arr = p_arr[rand_in] + np.dot(np.diag(z),(p_arr-p_arr[rand_in]))
#       
#       #calculate log post for proposal steps
#       for q in range(n): #calculate the log posteriors for all n
#         L_prop[q] = LogPosterior(pt_arr[q],*post_args)
#       
#       #accept or reject each step and assign to p and L_acc
#       acc_arr = np.where(np.random.rand(n) < (z**np.float(D-1.) * np.exp(L_prop - L_acc)))
#       p_arr[acc_arr] = pt_arr[acc_arr]
#       L_acc[acc_arr] = L_prop[acc_arr]
#       ParArr[i*n:(i+1)*n],PostArr[i*n:(i+1)*n] = p_arr,L_acc
                
    ####### end individual chain ###########
    PrintBar(n_ch,chain,(i+1)*n-1,ch_len*n,start,AccArr); print
    np.save(chain+".npy",np.concatenate([PostArr.reshape(PostArr.size,1),ParArr],axis=1))
    #np.savetxt(chain,np.concatenate([PostArr.reshape(PostArr.size,1),ParArr],axis=1))
  
  ####### end loop over chains ############
  print '-' * 80
  
##########################################################################################

def get_z(n,a):
  """get random draws from g(z) ~ z^-0.5 from 1/a to a"""
  x = np.random.rand(n) * (np.sqrt(4.*a)-np.sqrt(4./a)) + np.sqrt(4./a)
  return x**2 / 4.

##########################################################################################

def PrintBar(n,chain,i,ch_len,start,AccArr):
  "Print the status bar - probably a more elegant way to write this..."
  ts = time.time()-start
  a_str = "" if i <= ch_len/5 else ", acc = %.2f%%" % (100.*np.float(AccArr[ch_len/5:i].sum())/(i-ch_len/5+1))
  print "\rComputing Chain %d: '%s' %-20s t = %dm %.2fs%s " % (n+1,chain,'#'*(i/(ch_len/20)+1),ts // 60., ts % 60.,a_str),
  sys.stdout.flush();

##########################################################################################

def PrintParams(ch_filenames,n,ch_len,posterior,gp,ep):

  print "Infer.AffInvMCMC running..."
  print "MCMC parameters:"
  print " No Chains: %d" % len(ch_filenames)
  print " No walkers: %d" % n
  print " No free parameters: %d" % (np.array(ep)>0).sum()
  print " Chain Length: %d" % ch_len
  print " Posterior evaluations: %d" % (ch_len*n)
  print " Computing chains:", ch_filenames
  print " Posterior probability function: ", posterior
  print " Function params <value prop_size>:"
  for q in range(len(gp)):
    print "  p[%d] = %f +- %f" % (q,gp[q],ep[q])

##########################################################################################
