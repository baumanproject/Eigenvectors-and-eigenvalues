#pragma once

using namespace std;

#define CHECKER false

#ifdef MATHFUNCSDLL_EXPORTS
#define MATHFUNCSDLL_API __declspec(dllexport) 
#else
#define MATHFUNCSDLL_API __declspec(dllimport) 
#endif

class gpu_solver {

public:

	MATHFUNCSDLL_API double* GPU_gradient_solver(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Preconditioned conjugate gradient method
	MATHFUNCSDLL_API double* GPU_stab_bi_gradient_solver(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Biconjugate gradient stabilized method
	MATHFUNCSDLL_API double* GPU_stab_bi_gradient_solver_with_preconditioner(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size);//Preconditioned Biconjugate gradient stabilized biconjugate
	MATHFUNCSDLL_API void matrix_eigenvalues(double *val, int *col, int *row, double *diag, int non_zero, int size, double * eigen_values, double** eigen_vectors,double *v, double *val2, int *col2, int *row2, double *diag2, int non_zero2, int size2, int amount_ev);
};