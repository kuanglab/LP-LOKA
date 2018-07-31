#include "../includes.h"
#include "DisplayTools.h"

namespace BigLP {
namespace Tools {

bool DisplayTools::verbose = false;

void DisplayTools::PrintVector(const string &label, const vector<double> &v, int n) {
    cout << label << ": " << endl;
    for (int i=0; i<n; i++) cout << v[i] << " ";
    cout << endl;
}

void DisplayTools::PrintVector(const string &label, double *v, int n) {
    cout << label << ": ";
    for (int i=0; i<n; i++) cout << v[i] << " ";
    cout << endl;
}

void DisplayTools::PrintVector(const string &label, int *v, int n) {
    cout << label << ": ";
    for (int i=0; i<n; i++) cout << v[i] << " ";
    cout << endl;
}

void DisplayTools::PrintMatrix(const string &label, double *A, int m, int n) {
	cout << label << ": " << endl;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << A[i*n+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void DisplayTools::PrintMatrix(const string &label, const vector<vector<double> > &A, int m, int n) {
    cout << label << ": " << endl;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

} // namespace DataPreparation
} // namespace BigLP