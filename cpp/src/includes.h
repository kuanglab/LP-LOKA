#ifndef _INCL_HEADER_H_
#define _INCL_HEADER_H_

/*************************************************************************
* Header file inclusion section
**************************************************************************/
#include <iostream>
#include <iomanip>
#include <array>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include "mkl.h"
#ifdef INCLUDE_MPI
	#include <mpi.h>
#endif
#ifdef INCLUDE_OPENMP
	#include <omp.h>
#endif

/*************************************************************************
* Namespaces being used
**************************************************************************/
using namespace std;

#endif
