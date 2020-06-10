
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//define function prototype
double LevinsonTrenchZoharSolve(double *r, double *x, double *y, int N);

/*****************************************************************************************
Python code to implement the algorithm
Specific to symmetric Toeplitz matrix, but algorithm can be generalised

def LevinsonSymToe(r,y):
  #The Levinson-Trench-Zohar Algorithm for Exactly Toeplitz Matrices
  
  # loop over n = 0...N
  # r_n fixed, n=0-n
  # a,b,x,y are vectors of length n
  
  #initialise the variables
  N = y.size  
  a = np.zeros(N); a[0] = 1.
  e = r[0] # e is a scalar
  x = np.zeros(N)
  x[0] = y[0] / r[0]
  loge = np.log(e)
  
  #loop over n from n=1 (have the n=0 solution)
  for n in range(1,N):
    #calculate eta and v
    eta = - (1./e) *  np.sum(r[1:n+1][::-1]*a[:n])
    #v =   - (1./e) *  np.sum(r[1:n+1]*a[:n][::-1])
    
    #calculate a (b is just a reflection of a)
    a[:n+1] = a[:n+1] + eta*a[:n+1][::-1]
    
    #get new value of e
    e = e*(1.-eta**2)
    loge += np.log(e)
    
    #get lambda
    lmba = y[n] - np.sum(r[1:n+1][::-1]*x[:n])
    
    #finally calculate new x
    x[:n+1] = x[:n+1] + (lmba / e) * a[:n+1][::-1]
    
  return loge,x
*****************************************************************************************/

double LevinsonTrenchZoharSolve(double *r, double *x, double *y, int N)
//calculates the linear solution of Ax = b; ie x = A^-1b for symmetric Toeplitz matrix A
//and vector b using the Levenson-Trench-Zohar algorithm, complexity ~O(n^2). Also returns
//the log determinant required for calculation of Gaussian likelihood.
{
  double *a,e,log_det,xi,lmba,a_temp;
  int n,j;
  
  //set starting points for iteration
  a = malloc(N*sizeof(double));
  a[0] = 1.;
  e = r[0];
  x[0] = y[0] / r[0];
  
  log_det = log(e);
  
  //loop over n from n=1 (ie start with the n=0 solution,N-1 will yield the solution)
  for (n=1;n<N;n++){
	//sum over j=0-(n-1) to calculate xi (mu is the same for symmetric matrix)
    xi = 0;
    for (j=0;j<n;j++){
    	xi -= (r[n-j] * a[j]);
    	//v -= r[j+1] * a[n-j];
    	}
    xi /= e;
    
    //calculate a (b is just a reflection of a)
    a[n] = 0.; //need to be clever and do 2 at a time as a[n] depends on a[n-1]
	  for (j=0;j<=(n)>>1;j++){
      a_temp = a[j];
      a[j] += (xi * a[n-j]);
      if (j!=(n-j)) a[n-j] += (xi * a_temp);
      }

    //#get new value of e and calculate the log determinant
    e = e*(1.-(xi*xi));
    log_det += log(e);
    
    //calculate lambda
    lmba = 0;
    for (j=0;j<n;j++){
    	lmba += (r[n-j] * x[j]);
    	}
    lmba = y[n] - lmba;
    
    //#finally calculate new values of x
    x[n] = 0.;
    for (j=0;j<=n;j++){
    	x[j] += (lmba / e) * a[n-j];
    	}

    } //end loop over n
  
  free(a);
  
  return log_det;
}
