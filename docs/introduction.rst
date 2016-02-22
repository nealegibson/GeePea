
.. _overview:

Overview
========

This module was originally developed to fit transit light curves using
Gaussian Processes (GPs) in order to model the systematics as a stochastic process, as
described in `Gibson et al. (2012)`_, but it is also useful as a general tool for
fitting datasets using GPs with arbitrary kernel and mean functions. It was originally
part of my general Infer module, but new features have regularly been added, some of which are described
in more recent papers, see e.g. Gibson et al. (`2013a`_, `2013b`_) and `Gibson (2014)`_
for details. Please consider citing these if you make use of this code.

For an introduction to GPs please see these papers, or for a text book introduction see
`Gaussian Processes for Machine Learning`_. For a more generic introduction to Bayesian
inference including an relatively easy intro to GPs I recommend `Pattern Recognition and
Machine Learning`_. I also found this `tutorial`_ by M. Ebden incredibly useful when
learning GPs.

.. note::

  The standard solver uses cholesky decomposition (via scipy). I experimented with multiple methods
  (including C code, GPUs, other python implementations), and this was the
  fastest and most convenient solver I could find without using approximations (although
  GPUs are faster for large matrices). There is also a fast solver for Toplitz covaraince kernels
  (diagonal-constant matrix) using the Levenson-Trench-Zohar algorithm (C implementaion),
  and for convenience, a white noise kernel. The GP class will recognise the
  correct kernel-type for the built-in kernels. I might include additional solvers in the future,
  but these have proved sufficient for the light curve models I've explored so far.

.. _Gibson et al. (2012): http://adsabs.harvard.edu/abs/2012MNRAS.419.2683G
.. _2013a: http://adsabs.harvard.edu/abs/2013MNRAS.428.3680G
.. _2013b: http://adsabs.harvard.edu/abs/2013MNRAS.436.2974G
.. _Gibson (2014): http://adsabs.harvard.edu/abs/2014MNRAS.445.3401G
.. _Gaussian Processes for Machine Learning: http://www.gaussianprocess.org/gpml/
.. _Pattern Recognition and Machine Learning: http://www.springer.com/us/book/9780387310732
.. _tutorial: http://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf

Installation
============

Download the module from github at https://github.com/nealegibson/GeePea or clone using git::

  $ git clone https://github.com/nealegibson/GeePea

Then install as normal::

  $ cd GeePea
  $ python setup.py build
  $ python setup.py install [--prefix=/path_to_install_dir]

