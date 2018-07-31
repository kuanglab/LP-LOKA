namespace BigLP {
namespace Tools {

// Preprocess data for parallel processing
class DisplayTools {
    private:
    	
    public:
    	static bool verbose;
    	static void PrintVector(const string &label, const vector<double> &v, int n);
    	static void PrintVector(const string &label, double *v, int n);
    	static void PrintVector(const string &label, int *v, int n);
    	static void PrintMatrix(const string &label, double *A, int m, int n);
    	static void PrintMatrix(const string &label, const vector<vector<double> > &A, int m, int n);
};

} // namespace ML
} // namespace BigLP