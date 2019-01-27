#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <cmath>
#include <ctime>
#include <locale.h>
#include <iostream>
#include <omp.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <iomanip>
#define M1 false;
#define M2 false;
#define M3 false;
#define EIGEN true;
#include "mmio_wrapper.h"
#include "GPU.h"
using namespace std;

double* GPU_solve(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size,int method)
{

	gpu_solver obj;
	if (method == 1)
	{
		return obj.GPU_gradient_solver(val, col, row, right, diag, non_zero, size);
	}
		
	if (method == 2)
	{
		return obj.GPU_stab_bi_gradient_solver(val, col, row, right, diag, non_zero, size);
		
	}
	if (method == 3)
	{
		return obj.GPU_stab_bi_gradient_solver_with_preconditioner(val, col, row, right, diag, non_zero, size);

	}
	
}

void eigen_values_and_vectors(double *val, int *col, int *row, double *diag, int non_zero, int size, double * eigen_values, double** eigen_vectors,double *right, double *val2, int *col2, int *row2, double *diag2, int non_zero2, int size2,int amount_ev)
{
	gpu_solver obj;
	obj.matrix_eigenvalues(val, col, row, diag, non_zero, size, eigen_values, eigen_vectors,right,val2,col2,row2,diag2,non_zero2,size2,amount_ev);

}

int main()
{
	/*omp_set_num_threads(4);
#pragma omp parallel for 
	for (int i = 0;i < omp_get_num_threads();i++)
	{ 
	#pragma omp critical
		cout << "Hello! It should be thread No."<< i <<" but it is "<< omp_get_thread_num() << endl;
	}
*/
	int n;
	int m;
	int nnz;
	double *val;
	int *col;
	int *row;
	//loadMMSparseMatrix("bcsstk01.mtx", 'd', true, &m, &n, &nnz, &val, &row, &col);
	loadMMSparseMatrix("test.mtx", 'd', true, &m, &n, &nnz, &val, &row, &col);
	//test
	//nasa4704
	//bcsstm09
	//Kuu					BCG: 2.4 s			CG:		
	//af_shell3				BCG: 3.4 s          CG:
	// Hook_1498       BCG:  9.8 s		CG:
	// audikw_1     BSG 76.29  step 1361	 CG: 
	cout << "File uploaded!!! NOW COMPUTATION!";
	/*for (int i = 0; i < nnz; i++)
	{
		cout << val[i] << " " << col[i] << " " << row[i] << endl;
	}*/
	double* diag = new double[n];
	int q = 8;				// amount of first eigenvalues
	double *lambda_value = new double[q];
	double** lambda_vec = new double*[q];
	for (int i = 0; i < q; i++)
	{
		lambda_vec[i] = new double[n];
	}
	double* right = new double[n];
	double* sol1 = new double[n];
	for (int i = 0; i < n; i++)
	{
		right[i] = 0;
	}
	right[0] = 1;
	//right[1] = 0.046;
	//right[2] = 0.0872;
	//right[3] = -0.026;
	/*right[4] = -0.035;
	right[5] = -0.043;
	right[6] = -0.0534;
	right[7] = -0.016;
	right[8] = -0.071;*/
	right[9] = 1;
	if (row[0] == 1)
	{
		for (int i = 0; i < nnz; i++)
		{
			col[i] = col[i] - 1;
		}
		for (int i = 0; i <= n; i++)
		{
			row[i] = row[i] - 1;
		}

	}
	for (int i = 0; i < n; i++)
	{
		int k = row[i]; //  
		int r = 0;//смещение 
		while (col[k + r] != i)
		{
			r++;
		}
		diag[i] = val[k + r];
		val[k + r] = 0;
	}

	/*cout << "After transform\n";
	for (int i = 0; i < nnz; i++)
	{
		cout << val[i] << " " << col[i] << " " << row[i] << endl;
	}

	cout << "Diag:\n";
	for (int i = 0; i < n; i++)
	{
		cout << diag[i] << " ";
	}*/
#if M1
	clock_t int11 = clock();

	//Solving
	cout << endl << "CG:" << endl;
	sol1 = GPU_solve(val, col, row, right, diag, nnz, n, 1);
	clock_t int22 = clock();
	cout << endl << "TIME:  " << double(int22 - int11) / 1000.0 << endl;

	for (int i = 0; i < 10; i++)
	{
		cout << sol1[i] << " ";
	}
#endif

#if M2
	clock_t int1 = clock();

	//Solving
	cout << endl << "BiCGSTAB:" << endl;
	sol1 = GPU_solve(val, col, row, right, diag, nnz, n,2);
	clock_t int2 = clock();
	cout<<endl << "TIME:  " << double(int2 - int1) / 1000.0 << endl;

	for (int i = 0; i < 10; i++)
	{
		cout << sol1[i] << " ";
	}
	//cout << m << "  " << endl;
#endif
#if M3
	clock_t int3 = clock();
	cout << endl << "PBiCGSTAB:" << endl;
	sol1 = GPU_solve(val, col, row, right, diag, nnz, n,3);
	clock_t int4 = clock();
	cout << endl << "TIME:  " << double(int4 - int3) / 1000.0 << endl;

	for (int i = 0; i < 10; i++)
	{
		cout << sol1[i] << " ";
	}
#endif
#if EIGEN
	int n_2;
	int m_2;
	int nnz_2;
	double *val_2;
	int *col_2;
	int *row_2;
	//loadMMSparseMatrix("bcsstk01.mtx", 'd', true, &m, &n, &nnz, &val, &row, &col);
	loadMMSparseMatrix("testM.mtx", 'd', true, &m_2, &n_2, &nnz_2, &val_2, &row_2, &col_2);
	//nasa4704
	//testM
	//bcsstm09
	//Kuu					BCG: 2.4 s			CG:		
	//af_shell3				BCG: 3.4 s          CG:
	// Hook_1498       BCG:  9.8 s		CG:
	// audikw_1     BSG 76.29  step 1361	 CG: 
	cout << "File uploaded for matrix M!!! NOW COMPUTATION!";
	/*for (int i = 0; i < nnz; i++)
	{
	cout << val[i] << " " << col[i] << " " << row[i] << endl;
	}*/
	double* diag_2 = new double[n_2];
	/*double *lambda_value_2 = new double[n_2];
	double** lambda_vec_2 = new double*[n_2];
	for (int i = 0; i < n_2; i++)
	{
		lambda_vec_2[i] = new double[n_2];
	}*/
	double* right_2 = new double[n_2];
	//double* sol1_2 = new double[n_2];
	for (int i = 0; i < n_2; i++)
	{
		right_2[i] = 1;// rand() % 10;
	}

	if (row_2[0] == 1)
	{
		for (int i = 0; i < nnz_2; i++)
		{
			col_2[i] = col_2[i] - 1;
		}
		for (int i = 0; i <= n_2; i++)
		{
			row_2[i] = row_2[i] - 1;
		}

	}
	for (int i = 0; i < n_2; i++)
	{
		int k_2 = row_2[i]; //  
		int r_2 = 0;//смещение 
		while (col_2[k_2 + r_2] != i)
		{
			r_2++;
		}
		diag_2[i] = val_2[k_2 + r_2];
		val_2[k_2+ r_2] = 0;
	}
	clock_t int5 = clock();
	cout << endl << "Eigen_values:" << endl;
	eigen_values_and_vectors(val,col,row,diag,nnz,n,lambda_value,lambda_vec,right,val_2,col_2,row_2,diag_2,nnz_2,n_2,q);
	//sol1 = GPU_solve(val, col, row, right, diag, nnz, n, 3);
	clock_t int6 = clock();
	cout << endl << "TIME:  " << double(int6 - int5) / 1000.0 << endl;
	cout << endl << "Received Eigenvalues:" << endl;
	for (int i = 0; i < q; i++)
	{
		cout <<"Value["<<i<<"] = "<< std::setprecision(17)<<lambda_value[i] << endl;
	}

#endif
	system("pause");


	return 0;
}