
GPs in a nutshell
-----------------

A Gaussian Process (GP) is a non-parametric method for regression, used extensively for
regression and classification problems in the machine learning community. Please refer to the
papers and textbooks mentioned in the :ref:`overview` for a full introduction to GPs.

A GP is formally defined as an collection of random variables :math:`\by`, any finite number of
which have a joint Gaussian distribution, e.g.:

.. math::

  \by \sim \N(\bmu, \Bsig)

Generally speaking, when we fit a model :math:`T` or *mean function* with parameters :math:`\ph` to
data :math:`\X` and :math:`\by`, collectively refered to as the data, :math:`D`, we construct
a likelihood function under the assumption of independent Gaussian uncertainties, :math:`\sigma_i`:

.. math::

	\L(\D | \ph) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp \left( -\frac{ (y_i - T(\bx_i,\ph))^2 }{2\sigma_i^2} \right)\\

This likelihood is central to inferring the probabilty distributions of our parameters :math:`\ph`.
We can optimise it with respect to :math:`\ph`, or construct a posterior distribution for :math:`\ph`
via Bayes theorem after defining priors. Note that the term inside the exponential is :math:`-\frac{1}{2}\chi^2`,
so optimising the likelihood is equivalent to minimising :math:`\chi^2` if the uncertainties are held
fixed. This is already a Gaussian process, albeit a trivial one.

More generally, we can use a multi-variate Gaussian distribution as our likelihood, where we consider covariance
between the data points:

.. math::

  \L(\D | \th, \ph) = \frac{1}{|2\pi\Sigma|^{1/2}} \exp\left( -\frac{1}{2}\br^T\Bsig^{-1}\br \right),

where in addition to a mean function we define a kernel function to populate the covariance matrix:

.. math::
  \Bsig_{ij} = k(\bx_i,\bx_j,\th)

and have defined :math:`\br = \by - T(\X,\ph)`. The most commonly used kernel, and the default
in this code, is the squared exponential kernel:

.. math::

  k(\bx_i,\bx_j,\th) = \xi^2 exp\left( - \sum_{k=1}^K \frac{(x_{ik} - x_{jk})^2}{2l_k^2} ) \right)
    + \delta_{ij}\sigma^2

where :math:`\th = \{\xi,\boldsymbol{l},\sigma\}`. Here :math:`\xi` represents the *height scale*,
:math:`\boldsymbol{l}` is the vector of *length scales* (one for each input dimension of
:math:`\X`), and :math:`\sigma` is the white noise term.

This kernel states that data points near each other in input space are highly correlated, and
those far away from each other are poorly correlated, so we can learn a lot about our underlying
function space near observed data points. The squared exponential kernel defines an infinitely
smooth function space.

.. note::
  The parameters of both the kernel and the mean function are now referred
  to as *hyperparameters* of the model.

This likelihood function can be treated in exactly the same way as the previous one,
ie. you can optimise with respect to all of the hyperparameters, marginalise over it, etc. This
documentation will provide examples of how to do this with GeePea. Now lets get started with
:ref:`a simple example <getting started>`.