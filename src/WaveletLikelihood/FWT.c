
#include "FWT.h"

/********************************************************************
Performs fast wavelet transform on data of size 2^n (n pos integer).
Data is replaced by its transform.
********************************************************************/
int DoFWT(double *data, size_t size)
{
	//check if size is a power of 2
	if(!IsPow2(size)){
		fprintf(stderr, "DoFWT Error: data for DoFWT is not a power of 2. Exiting...\n");
		exit(1);}
	
	//define gsl structures
	gsl_wavelet *wavelet;
	gsl_wavelet_workspace *workspace;
	
	//assign wavelet type
	wavelet = gsl_wavelet_alloc(gsl_wavelet_daubechies, 4);
	
	//assign workspace
	workspace = gsl_wavelet_workspace_alloc(size);

	//perform wavelet transform - check for success
	if((gsl_wavelet_transform_forward (wavelet, data, 1, size, workspace)) != GSL_SUCCESS){
		fprintf(stderr, "DoFWT Error: forward transfer failed.");
		exit(1);
		}

	//free wavelet
	gsl_wavelet_free(wavelet);

	//free workspace
	gsl_wavelet_workspace_free(workspace);	
	
	return 1;
}

/********************************************************************
Performs inverse fast wavelet transform on data of size 2^n (n pos integer).
Data is replaced by its transform.
********************************************************************/
int DoIFWT(double *data, size_t size)
{
	//check if size is a power of 2
	if(!IsPow2(size)){
		fprintf(stderr, "DoIFWT Error: data for DoFWT is not a power of 2. Exiting...\n");
		exit(1);}
	
	//define gsl structures
	gsl_wavelet *wavelet;
	gsl_wavelet_workspace *workspace;
	
	//assign wavelet type
	wavelet = gsl_wavelet_alloc(gsl_wavelet_daubechies, 4);
	
	//assign workspace
	workspace = gsl_wavelet_workspace_alloc(size);

	//perform wavelet transform - check for success
	if((gsl_wavelet_transform_inverse (wavelet, data, 1, size, workspace)) != GSL_SUCCESS){
		fprintf(stderr, "DoIFWT Error: inverse transfer failed.");
		exit(1);
		}

	//free wavelet
	gsl_wavelet_free(wavelet);

	//free workspace
	gsl_wavelet_workspace_free(workspace);	
	
	return 1;
}

/********************************************************************
Returns 1 if input is a power of 2
********************************************************************/
int IsPow2(size_t n)
{
	//while x is even and greater than 1, divide by two, will get 1 if n == 2^x
	while( ((n % 2) == 0) && (n > 1) ) n /= 2;
	return (n == 1);
}

/********************************************************************
********************************************************************/
