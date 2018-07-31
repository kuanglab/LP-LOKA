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

    // function variables
    string filename, outputPrefix;
    int n, k, nbyproc, precision = 6;
    string line;
    double x;
    double* Cp; // C partition of process p

    // read parameters
    for (int i=0; i<argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            filename = argv[i+1];
        }
        if (strcmp(argv[i], "-n") == 0) {
            n = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "-k") == 0) {
            k = atoi(argv[i+1]);
        }
        if (strcmp(argv[i], "--verbose") == 0) {
            DisplayTools::verbose = true;
        }
        if (strcmp(argv[i], "-o") == 0) {
            outputPrefix = argv[i+1];
        }
        if (strcmp(argv[i], "-p") == 0) {
            precision = atoi(argv[i+1]);
        }
    }

    nbyproc = ceil(1.0*n/nprocs);
    // make sure W is located in p0
    if (nbyproc < k) {
        cerr << "Error: Make sure ceil(n/np) > k \n";
        exit(1);
    }

    // read C
    Cp = new double[(lploka_int)nbyproc*k]();

    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Reading C..." << endl;
    }
    
    ifstream fileIN;
    fileIN.open(filename);
    
    if(fileIN.fail()){
        cerr << "Error: The C file you are tring to access cannot be found or open \n";
        exit(1);
    } else {
        int i=0;
        while(getline(fileIN,line)){
            istringstream streamA(line);
            int j = 0;
            while(streamA >>x){
                if ((i/nbyproc) == mpirank) {
                    Cp[(i%nbyproc)*k + j] = x;
                }
                j++;
                if (j >= k) break;
            }
            i++;
            if (i >= n) break;
        }
    }
    fileIN.close();

    // finished reading C
    MPI_Barrier(MPI_COMM_WORLD);
    if (DisplayTools::verbose && mpirank == 0) {
       cout << "Finished reading C..." << endl;
    }
    
    // run Nystrom
    Nystrom *nystrom = new Nystrom(mpirank, nprocs);
    double *Fp = nystrom->performParallel(Cp, n, k);

    // write output
    if (!outputPrefix.empty()) {
        ofstream outfile;
        string filepath = outputPrefix + to_string(mpirank) + ".txt";
        outfile.open(filepath);
        for (int i=0; i<nbyproc; i++) {
            for (int j=0; j<k; j++) {
                outfile << std::setprecision(precision) << Fp[i*k+j] << " ";
            }
            outfile << endl;
        }
        outfile.close();
    }


    MPI_Finalize();
    return 0;
}
