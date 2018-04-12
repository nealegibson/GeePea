#ifndef __FWT_H__
#define __FWT_H__

//include files
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_wavelet.h>

//function prototypes
int DoFWT(double *data, size_t size);
int DoIFWT(double *data, size_t size);
int IsPow2(size_t n);

#endif /* __FWT_H__ */
