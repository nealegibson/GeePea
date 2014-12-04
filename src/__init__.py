"""
Infer module - useful tools for inference. MCMC methods, optimisation, Importance
Sampling, GPs, etc, with special attention to make the code fast but adaptable. Much of it
is tested against equivalent C modules - it is almost as fast but much more flexible.

Neale Gibson
ngibson@eso.org
nealegibby@gmail.com

leastsqbound.py was adapted from https://github.com/jjhelmus/leastsqbound-scipy/
- It is Copyright (c) 2012 Jonathan J. Helmussee, see file for full license

"""

from MCMC import MCMC
from MCMC_SimN import MCMC_N
from MCMC_utils import *
from ImportanceSampling import *
from Conditionals import *
from Optimiser import *
from LevenbergMarquardt import *

from MCMC_BGibbs import *
from AffInv_MCMC import *

from InferGP import GP
from InferMGP import MGP

from GPUtils import *
import GPRegression as GPR
import GPCovarianceMatrix as GPC

from GPKernelFunctions import *
from GPPeriodicKernelFunctions import *

from GPToeplitz import *
