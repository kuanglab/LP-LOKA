namespace BigLP {
namespace ML {

// Performs Nystrom
class Nystrom {
    private:
    	int mpirank;
    	int nprocs;
    	int nbyproc;
    public:
      string indicesFilename;
    	Nystrom();
    	Nystrom(int mpirank, int nproc);

      double* performParallel(double *data, int n, int k);
};

} // namespace ML
} // namespace BigLP
