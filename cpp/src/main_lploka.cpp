#include "includes.h"
#include "types.h"
#include "ml/ml.h"
#include "tools/tools.h"

using namespace BigLP::ML;
using namespace BigLP::Tools;

int main(int argc, char **argv){
    int mpirank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    string pathF, pathf0, outputPrefix;
    int n, k, startIndex, endIndex, maxIter = 1000, saveRows, nbyproc, rows, precision = 6;
    double alpha=0.1, tol = 1e-9, x;
    double * Fp;
    string line;

    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-F") == 0) {
            pathF = argv[i+1];
        }
        if (strcmp(argv[i], "-f0") == 0) {
            pathf0 = argv[i+1];
        }
        if (strcmp(argv[i], "-n") == 0) {
            n = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-k") == 0) {
            k = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-startIndex") == 0) {
            startIndex = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-endIndex") == 0) {
            endIndex = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-o") == 0) {
            outputPrefix = argv[i+1];
        }
        if (strcmp(argv[i], "-maxIter") == 0) {
            maxIter = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-tol") == 0) {
            tol = atof(argv[i+1]);
        }
        if (strcmp(argv[i], "--verbose") == 0) {
            DisplayTools::verbose = true;
        }
        if (strcmp(argv[i], "-alpha") == 0) {
            alpha = atof(argv[i+1]);
        }
        if (strcmp(argv[i], "-saveRows") == 0) {
            saveRows = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-p") == 0) {
            precision = atoi(argv[i+1]);
        }
    }

    nbyproc = ceil(1.0*n/nprocs);
    rows = nbyproc*nprocs;

    if (saveRows == 0) {
        saveRows = rows;
    }

    // read F
    Fp = new double[nbyproc*k]();

    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Reading F..." << endl;
    }

    ifstream file;
    file.open(pathF);
    
    if(file.fail()){
        cerr << "Error: The F file you are tring to access cannot be found or open \n";
        exit(1);
    } else {
        int i=0;
        while(getline(file,line)){
            istringstream streamA(line);
            int j = 0;
            while(streamA >>x){
                if ((i/nbyproc) == mpirank) {
                    Fp[(i%nbyproc)*k + j] = x;
                }
                j++;
                if (j >= k) break;
            }
            i++;
            if (i >= n) break;
        }
    }
    file.close();

    // finish reading F
    MPI_Barrier(MPI_COMM_WORLD);
    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Finished reading F..." << endl;
    }

    // start normalization

    // sum(F)
    double * Fsump = new double[k]();
    for (lploka_int i=0; i<nbyproc; i++) {
        for (lploka_int j=0; j<k; j++) {
            Fsump[j] += Fp[i*k+j];
        }
    }
    double * Fsum = new double[k]();
    MPI_Allreduce(Fsump, Fsum, k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    delete[] Fsump;

    // Ssum = F*sum(F)'
    double * Ssum = new double[nbyproc];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, nbyproc, k, 1, Fp, k, Fsum, 1, 0, Ssum, 1);
    delete[] Fsum;

    // Fp = Fp ./ sqrt(abs(Ssum))
    for (lploka_int i=0; i<nbyproc; i++) {
        for (lploka_int j=0; j<k; j++) {
            if (Ssum[i] == 0) {
                Fp[i*k+j] = 0;
            } else {
                Fp[i*k+j] /= sqrt(abs(Ssum[i]));
            }
        }
    }
    delete[] Ssum;
    // finish normalization

    // do query by query
    for (int q=startIndex; q < endIndex; q++) {
        if (DisplayTools::verbose && mpirank == 0) {
    	   cout << "query: " << q << endl;
        }

        // initialize f
        double *finit = new double[nbyproc]();
        for (int i=0; i < nbyproc; i++) {
            finit[i] = 1.0/rows;
        }

        // f0
        double *f0 = new double[nbyproc]();
        file.open(pathf0 + "/" + to_string(q+1) + ".txt");

        if(file.fail()){
            cerr << "Error: The f0 file you are tring to access cannot be found or open \n";
            exit(1);
        }

        if(file.good()){
            int i=0;
            while(getline(file,line)){
                istringstream streamA(line);
                while(streamA >>x){
                    if ((i/nbyproc) == mpirank) {
                        f0[i%nbyproc] = x;
                    }
                    i++;
                }
                if (i >= n) break;
            }
        }
        file.close();    

        // run LRLP
        LowRankLP *lowRankLP = new LowRankLP(mpirank, nprocs);
        double *f = lowRankLP->performParallel(Fp,f0,finit,rows,k,alpha,maxIter,tol);

        if (!outputPrefix.empty()) {
            for (int i=0; i < nprocs; i++) {
                if (i == mpirank) {

                    ofstream outfile;
                    string filepath = outputPrefix + "/" + to_string(q+1) + ".txt";
                    outfile.open(filepath, ios::app);
                    for (int j=0; j<nbyproc; j++) {
                        if (mpirank*nbyproc + j < saveRows) {
                            outfile << std::setprecision(precision) << f[j] << endl;
                        }
                    }
                    outfile.close();
                    
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        
        delete[] f0;

    }

    MPI_Finalize();
    
    
    return 0;

}


