#!/usr/bin/env python

import numpy as np
import scipy.linalg as LA
import ctypes
from numpy.ctypeslib import ndpointer
import os

#import the c function using ctypes
WaveletLogLikelihood_C = ctypes.CDLL('{}/WaveletLikelihood.so'.format(os.path.dirname(__file__))).WaveletLikelihood_C

#specify the argument and return types
#double WaveletLikelihood_C(double* array, int size, double sig_w, double sig_r, double gamma, int verbose)
WaveletLogLikelihood_C.argtypes = [ndpointer(ctypes.c_double),ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int]
WaveletLogLikelihood_C.restype = ctypes.c_double

def Wavelet(yerr,theta):
  """
  Special kernel for wavelet methods - identical to white noise with different attributes  
  """
  
  #Calculate distance matrix without scaling
  sig = theta[0] * yerr
  
  return sig**2

#add kernel attributes
Wavelet.n_par = 3
Wavelet.kernel_type = 'Wavelet'
WV = Wavelet # and alias
