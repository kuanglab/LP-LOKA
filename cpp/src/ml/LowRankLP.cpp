#include "../includes.h"
#include "../ml/ml.h"
#include "../tools/tools.h"

using namespace BigLP::Tools;

namespace BigLP {
namespace ML {

LowRankLP::LowRankLP() {
    mpirank = 0;
    nprocs = 0;
    nbyproc = 0;
}

LowRankLP::LowRankLP(int mpirank, int nprocs) {
    this->mpirank = mpirank;
    this->nprocs = nprocs;
}

double* LowRankLP::performParallel(double *Fp, double *f0, double *fp, int n, int rank, double alpha, int maxIter, double tol) {
    
    nbyproc = ceil(1.0*n/nprocs);

    MPI_Barrier(MPI_COMM_WORLD);

    // time starts here
    double MPIelapsed;
    double MPIt2;
    double MPIt1;
    MPIt1 = MPI_Wtime();

    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Low Rank LP: Starting main iterations..." << endl;
    }

    // iterarions loop
    for (int t = 0; t < maxIter; t++){
        double *fpold = new double[nbyproc];
        double *fptmp = new double[rank];
        double *ftmp = new double[rank];

        copy(fp, fp+nbyproc, fpold);
        
        // f_new = alpha*A*f_old + (1-alpha)*f0
        cblas_dgemv(CblasRowMajor, CblasTrans, nbyproc, rank, 1, Fp, rank, fp, 1, 0, fptmp, 1);
        MPI_Allreduce(fptmp, ftmp, rank, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        copy(f0, f0+nbyproc, fp);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, nbyproc, rank, alpha, Fp, rank, ftmp, 1, (1-alpha), fp, 1);
        
        // find local max elem diff
        double maxElemDiffp = 0;
        for (int i = 0; i < nbyproc; i++){
            if (abs(fp[i]-fpold[i])>maxElemDiffp) {
                maxElemDiffp = abs(fp[i]-fpold[i]);
            }
        }
        
        // find global max
        double maxElemDiff = 0;
        MPI_Allreduce(&maxElemDiffp, &maxElemDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (DisplayTools::verbose && mpirank == 0) {
           cout << "Low Rank LP: Iter " << t << " - Convergence max. diff.: " << maxElemDiff << endl;
        }
        if (maxElemDiff < tol  ) {
            break;
        }
        delete[] fpold;
        delete[] fptmp;
        delete[] ftmp;
    }

    MPIt2 = MPI_Wtime();
    MPIelapsed = MPIt2 - MPIt1;
    if(DisplayTools::verbose && mpirank == 0){
        cout << "Parallel Run Time: " << MPIelapsed << endl;
    }

    return fp;
    
}

} // namespace DataPreparation
} // namespace BigLP
