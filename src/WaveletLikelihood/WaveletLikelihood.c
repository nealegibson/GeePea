/*********************************************************************************
*********************************************************************************/

#include "WaveletLikelihood.h"

/********************************************************************************/
//function to compute markov chains given chain names, lightcurve filenames
//and all chain parameters
double WaveletLikelihood_C(double* array, int size, double sig_w, double sig_r, double gamma, int verbose)
{
	
	int n_w,m_w;
	double LogLikelihood, chi_sq, sigma_S, sigma_W;
	double *residuals;
	int size_n = MakePowerOf2(size); //get nearest higher power of 2
	int m_levels = LogTwo(size_n); //get number of levels for lightcurve
	
	//define the array for residuals (with zero padding)
	residuals = (double *) malloc (size_n*sizeof(double));
	
	//initialise the residuals array
	for (n_w=0;n_w<size;n_w++) *(residuals+n_w) = *(array+n_w);
	for (n_w=size;n_w<size_n;n_w++) *(residuals+n_w) = 0.;
		
 	if (verbose) printf("WL size, size_n, m_levels, sig_w, sig_r, gamma: %d %d %d %lf %lf %lf\n", size, size_n, m_levels, sig_w, sig_r, gamma);
	
	DoFWT(residuals, size_n); //do wavelet transform (in place!)
	
	LogLikelihood = 0.;
	chi_sq = 0.;
	
	//now calculate chi squared + loglikelihood from the n0*2^M-1 wavelet coeff (and n0 scaling coeff) - assuming n0=1
	//first term out of the sumation
	sigma_S = sqrt( SQ(sig_r) * pow(2.,-gamma) * GAMMA_G + SQ(sig_w) );
	LogLikelihood -= log(sqrt(2*M_PI*SQ(sigma_S)));
	chi_sq = SQ( *(residuals+0) / sigma_S); //scaling coefficient	
	//sum up the chi_sq and loglikelihood for the wavelet transformed residuals
	for(m_w=1;m_w<=m_levels;m_w++){ //wavelet coefficients/
		sigma_W = sqrt( SQ(sig_r) * pow(2.,-gamma*m_w) + SQ(sig_w) ); //calculate the noise term
		for(n_w=0;n_w<(int)pow(2,m_w-1);n_w++){
			chi_sq += SQ( *(residuals+(int)pow(2,m_w-1)+n_w) / sigma_W); //and wavelet coefficient contributions to chi_sq	
			LogLikelihood -= log(sqrt(2*M_PI*SQ(sigma_W))); //sum up const terms in loglikelihood
			}
		}
//	printf("test:chisq = %lf\n", chi_sq);
	LogLikelihood -= chi_sq/2.;

 	if (verbose) printf("WL chi_sq loglikelihood: %lf %lf\n", chi_sq, LogLikelihood);
    
    free(residuals);
    
	return LogLikelihood;
}
/********************************************************************************/
//function to return the next highest power of two after N
size_t MakePowerOf2(size_t N)
{
	size_t x = 1;
	
	while(x<N) x*=2;
	
	return x;
}
/********************************************************************************/
//function to return the next highest power of two after N
int LogTwo(int N)
{
	int x = 0;
	
	while(N != 1){
		N/=2;
		x+=1;
		}
		
	return x;
}
/********************************************************************************/
