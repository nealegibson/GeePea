
Defining kernels
----------------

To select another kernel, this just needs to be passed to the GP class during
initialisation::

  >>> gp = GeePea.GP(x,y,p,kf=GeePea.Matern32)
  
or alternatively added using the GP.set_pars method::

  >>> gp.set_pars(kf=GeePea.Matern32)
  
This is important to do via the set_pars method as the number of parameters accepted by
the kernel function and the kernel type is automatically set for built-in kernels. This
would be ignored if the kernel function is set directly::

  >>> gp.kf = some_kernel
  
For a list of built in kernel functions see :ref:`kernels <kernels>` and :ref:`toeplitz kernels <toekernels>`.

It is also very easy to design your own kernels::

  def MySqExponential(X,Y,theta,white_noise=False):
    # This is just a copy of GeePea.SqExponential

    #Calculate distance matrix with scaling - multiply each coord by sqrt(eta)
    D2 = GeePea.EuclideanDist2(X,Y,v=1./(np.array(theta[1:-1])))#
    
    #Calculate covariance matrix
    K = theta[0]**2 * np.exp( - 0.5 * D2 )
	
    #Add white noise
    if white_noise is True: K += np.identity(X[:,0].size) * (theta[-1]**2)
	
    return np.matrix(K)
  
  # optional - add some attributes
  MySqExponential.n_par = lambda D: D+2
  MySqExponential.kernel_type = "Full"

Then the kernel can be added to the GP initialisation::

  >>> gp = GeePea.GP(x,y,p,kf=MySqExponential)
  # or later via:
  >>> gp = GeePea.GP(x,y,p)
  >>> gp.set_pars(kf=MySqExponential)

It is important that the function follows the same format, and accepts three input
arguments X, Y and theta, plus one additional boolean argument, white_noise. X and Y are
input matrices containing the arguments to the kernel, theta contains the parameters of
the mean function, and if white_noise is True, then the white noise term is added along
the diagonal of the covariance matrix. This format is required so that it can return
arbitrary blocks of the covariance matrix for GP regression, however, usually it is called
as::

  >>> kf(X,X,white_noise=True)
  
in order to calculate the full covariance matrix of the input data and calculate the
likelihood.

The attributes n_par and kernel_type are optional, and are used for the GP class to
automatically set the number of parameters the kernel takes and the type of kernel. These
can however be set directly via the GP class initialisation, via::

  >>> gp = GeePea.GP(x,y,p,kf=GeePea.Matern32,n_hp=3,kernel_type='F')
  
The kernel_type is 'Full' by default, so it only needs to be set if the kernel is reset
from a different type, or if Toeplitz/White kernels are defined by the user without a
kernel_type attribute.

