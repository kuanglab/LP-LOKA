#include "../includes.h"
#include "../ml/ml.h"
#include "../tools/tools.h"

using namespace BigLP::Tools;

namespace BigLP {
namespace ML {

Nystrom::Nystrom() {
    mpirank = 0;
    nprocs = 0;
    nbyproc = 0;
}

Nystrom::Nystrom(int mpirank, int nprocs) {
    this->mpirank = mpirank;
    this->nprocs = nprocs;
    this->nbyproc = 0;
}

double* Nystrom::performParallel(double *Cp, int n, int k) {

    nbyproc = ceil(1.0*n/nprocs);

    double MPIelapsed;
    double MPIt2;
    double MPIt1;
    MPIt1 = MPI_Wtime();

    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Fixing diagonal of W..." << endl;
    }

    // fix diagonal of W to make it positive definite (Gershgorin)
    double *c = new double[k]();
    double *cTotal = new double[k]();

    for (int i=0; i < nbyproc; i++) {
        for (int j=0; j < k; j++) {
            c[j] += Cp[i*k + j];
        }
    }
    
    MPI_Allreduce(c, cTotal, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (mpirank == 0) {
        for (int i=0; i < k; i++) {
            Cp[i*k+i] = cTotal[i];
        }
    }
    delete[] c;
    delete[] cTotal;

    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Computing eigenvectors of W..." << endl;
    }

    // computing eigenvectors and eigenvalues of W
    double *eigVals = new double[k];
    double *eigVectors = new double[k*k];
    
    if (mpirank == 0) {
        double wkopt; int lwork = -1; double* work; int info;
        char JOBZ[] = "V"; 
        char UPLO[] = "U";

        copy(Cp, Cp+k*k, eigVectors);
        dsyev_(JOBZ, UPLO, &k, eigVectors, &k, eigVals, &wkopt, &lwork, &info);
        lwork = (int)wkopt;
        work = new double[lwork];
        dsyev_(JOBZ, UPLO, &k, eigVectors, &k, eigVals, work, &lwork, &info);
        delete[] work;
    }

    // sending eigenvectors / eigenvalues to all processes
    MPI_Bcast(eigVectors, k*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(eigVals, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (DisplayTools::verbose && mpirank == 0) {
        cout << "Computing G..." << endl;
    }
    // F = C*U
    double *Fp = new double[nbyproc*k]();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nbyproc, k, k, 1, Cp, k, eigVectors, k, 0, Fp, k);
    delete[] eigVectors;

    // F = F*diag(eigVals)
    for (int i=0; i<nbyproc; i++) {
        for (int j=0; j<k; j++) {
            Fp[i*k+j] *= sqrt(1.0/eigVals[j]);
        }
    }
    delete[] eigVals;

    MPI_Barrier(MPI_COMM_WORLD);

    MPIt2 = MPI_Wtime();
    MPIelapsed = MPIt2 - MPIt1;
    if(DisplayTools::verbose && mpirank == 0){
        cout << "Parallel Run Time: " << MPIelapsed << endl;
    }

    return Fp;
}

} // namespace DataPreparation
} // namespace BigLP
