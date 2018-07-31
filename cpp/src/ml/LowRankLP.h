namespace BigLP {
namespace ML {

// Performs LowRankLP
class LowRankLP {
    private:
        int mpirank;
        int nprocs;
        int nbyproc;
        int color;
    public:
        LowRankLP();
        LowRankLP(int mpirank, int nproc);

        double* performParallel(double *Fp, double *f0, double *f, int n, int rank, double alpha, int maxIter, double tol);
};

} // namespace ML
} // namespace BigLP