
/********************************************************************************/

//#include <Python.h>
//#include <numpy/arrayobject.h> //for PyArray_Type objects
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "FWT.h"

#define GAMMA_G 0.72
#define SQ(x) ((x)*(x))

/*********************************************************************************
function prototypes
*********************************************************************************/

//module methods to be called from python
//static PyObject * WaveletLikelihood(PyObject *self, PyObject *args, PyObject *keywds);

//internal c functions
double WaveletLikelihood_C(double* array, int size, double sig_w, double sig_r, double gamma, int verbose);
size_t MakePowerOf2(size_t N);
int LogTwo(int N);
