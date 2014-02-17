
Python module to implement Bayesian inference routines.

Neale Gibson
ngibson@eso.org
nealegibby@gmail.com

This module has merged my MCMC routines and Gaussian processes modules. The MCMC now includes orthogonal stepping, affine invariant methods (emcee), and I've made many additions to the Gaussian Process classes to include multiplicative systematics (not fully tested), easy integration to the inference and optimisation tools, better plotting interfaces, and more kernels. I've also added various methods to get the Bayesian evidence, eg importance sampling - best initiated from MCMC samples. These codes or earlier variations have been used for the inference in my recent papers. Please contact me should you wish to use this code, I'm happy to discuss any modifications or suggestions.
