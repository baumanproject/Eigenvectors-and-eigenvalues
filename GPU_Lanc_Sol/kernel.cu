#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <cmath>
#include <ctime>
#include <locale.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <cusolverDn.h>
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_cusolver.h>
#include <device_launch_parameters.h>

#include "GPU.h"


#define SK_Nev 1.e-6
#define STEP_LIMIT 100000
#define APPROX 1.e-15
#define eps_for_b 1.e-6
//#define MAXITER 200000
//#define MAXRESIDUE 1.e-10
void Debuger_for_matr(double * input, int rows, int columns, double* checking);
void GPU_mult(double* vec, int size, int *nnz, double* diag, int gpu_amount, double **d_A, int **d_B, int ** d_C, double* rezult, int maximumThreads);
int* split(int gpu_amount, double* A, int* B, int* C, int size, int non_zero, double **d_A, int ** d_B, int **d_C);
//void Debuger(double* input, int size);
void Debuger(double * input, int size, double* checking);
void printMatrix(int m, int n, const double*A, int lda, const char* name);
void GPU_mult_for_little_gradient(double* vec, int size, int* nnz, double* diag, int gpu_amount, double** d_val, int** d_col, int** d_row, double* rezult);
void show_eigen_value(double*input, int lanc_count);
//double* GPU_stab_bi_gradient_solver_with_preconditioner(double *val, int *col, int *row, double *right, double *diag, int nnz, int size);

/*ret_val << <1, 1 >> > (converge_temp,i,tmp5);
checkCudaErrors(cudaMemcpy(tmp1, tmp5, sizeof(double), cudaMemcpyDeviceToHost));
ret_val << <1, 1 >> > (eigen_values_gpu,int(*tmp1),tmp6);
element << <1, 1 >> > (converge_eig_val, *CONVERGE_AMOUNT, *tmp6);*/

//__global__ void rand_vec(double* input,int size)
//{
//	int i = rand() % size;
//	input[i] = 1;
//}

__global__ void copy_v(double* input, int size,double*where)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	where[i] = input[i];
}

__global__ void add_to_converge_values(double* converge_temp,int i,double * eigen_values_gpu,double *converge_eig_val,int * CONVERGE_AMOUNT, double * tmp5)
{
	*tmp5 = converge_temp[i];
	converge_eig_val[(*CONVERGE_AMOUNT)+i] = eigen_values_gpu[int(converge_temp[i])];
}

__global__ void proverb(double* eigvecT, double *beta_q, int amount_ev,double eps,int * converge_amount,double * converge_val_number,double * converge_temp)
{
	/*int i = blockDim.x*blockIdx.x + threadIdx.x;*/
	/*if (i < amount_ev)*/
	for(int i=0;i<amount_ev;i++)
		if (abs(eigvecT[amount_ev*i + (amount_ev - 1)] * (*beta_q)) <= eps)
		{    
			 converge_val_number[int(*converge_temp)] = i;
			*converge_temp += 1;
			
			//*converge_amount += 1;
		}
		
}

__global__ void vec_mul_number(double* A, double value, int size, double* res)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		res[i] = A[i] * (value);
	}
}

__global__ void connect_diag_matr(double* matrix, double * a, double * b, int lanc_amount)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < lanc_amount)
	{
		matrix[i*lanc_amount + i] = a[i];
		if (i != lanc_amount - 1)
		{
			matrix[i*lanc_amount+i+1] = b[i];
			matrix[i*lanc_amount + i + lanc_amount] = b[i];
		}
	}

}

//__global__ void correlation(double* matrix, double *input_vector, int size, int count)
//{
//
//}

__global__ void reverse_for_eigen_values_lanc(double* input, int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		input[i] = 1 / input[i];
	}
}

__global__ void matr_add(double* main, int count, double *arr,int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
		main[count*size + i] = arr[i];
}

__global__ void return_vec(double* main, int count, int size,double * arr)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
		arr[i] = main[count*size + i];
}

__global__ void element(double* A, int i, double res)
{
	A[i] = res;
}

__global__ void ret_val(double* A, int i, double *res)
{
	*res = A[i];
}

__global__ void all_zero(int size, double* res)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		res[i] = 0;
	}
}


__global__ void vector_addition(double *input, int size,double* result)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		result[i] += input[i];
	}
}


__global__ void not_full_scalar(double* A, double* B, int size, double* res)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		res[i] = A[i] * B[i];
	}
}

__global__ void diag_revers(double* diag, double* res, int size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size)
	{
		res[i] = 1/diag[i];
	}
}

//void Debuger(double* input, int size,int checking)
//{
//	double * test = new double[size];
//	checkCudaErrors(cudaMemcpy(test, input, sizeof(double)*(size), cudaMemcpyDeviceToHost));
//	for (int i = 0; i<size; i++)
//	{
//		cout <<endl<< "test[" << i << "] = " << test[i];
//	}
//	cout << endl;
//	delete test;
//}

void Debuger(double * input, int size, double* checking)
{
	checkCudaErrors(cudaMemcpy(checking, input, sizeof(double)*(size), cudaMemcpyDeviceToHost));
}

void Debuger_for_matr(double * input, int rows, int columns, double* checking)
{
	checkCudaErrors(cudaMemcpy(checking, input, sizeof(double)*(rows)*(columns), cudaMemcpyDeviceToHost));
	for (int i = 0; i < columns; i++)
	{
		cout << "Vector " << "[" << i << "]" << endl;
		for (int j = 0; j < rows; j++)
		{
			cout << checking[i*rows + j] << " ";
		}
		cout << endl;
	}
}


void show_eigen_value(double*input,int lanc_count)
	{
		double* checking = new double[lanc_count];
		checkCudaErrors(cudaMemcpy(checking, input, sizeof(double)*(lanc_count), cudaMemcpyDeviceToHost));
		cout << endl << "Eigenvalues: " << endl;
		for (int i = 0; i < lanc_count; i++)
			cout << "Val ["<<i<<"] = "<<std::setprecision(17)<< checking[i] << endl;
		cout << endl;

	}




inline double dot_product(double* A, double* B, int size)
{
	double rezult = 0;
	for (int i = 0; i < size; i++)
	{
		rezult += (A[i] * B[i]);
	}
	return rezult;
}

inline void vector_on_number(double* A, double value, int size, double* res)
{
	for (int i = 0; i < size; i++)
		res[i] = A[i] * value;
}

inline void sum_vector(double* A, double* B, int size, double* res)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = A[i] + B[i];
	}
}

inline void raznost_vector(double* A, double* B, int size, double* res)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = A[i] - B[i];
	}
}

int return_string(int number, int* C)
{
	int i = 0;
	while (C[i] <= number)
		i++;
	return i;
}

int* split(int gpu_amount, double* val, int* col, int* row, int size, int non_zero, double **d_val, int ** d_col, int **d_row) // Êîñòëÿâî
{
	int mod = non_zero / gpu_amount; // óõîäèò íà âñå 
	int rest = non_zero - mod*(gpu_amount - 1); //óõîäèò íà ïîñëåäíþþ 
	int first_position;
	int last_position;
	int first_string;
	int last_string;
	double *val_;
	int *col_;
	int *row_;

	int *temp = new int[gpu_amount];
	int nsize;

#if CHECKER
	cout << endl << "CSR:" << endl;
	for (int i = 0; i < non_zero; i++)
	{
		cout << val[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < non_zero; i++)
	{
		cout << col[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < size + 1; i++)
	{
		cout << row[i] << " ";
	}
	cout << endl;
#endif

	for (int number = 0; number < gpu_amount; number++)
	{
		if (number == gpu_amount - 1)
		{
			int in1 = 0;
			int in2 = 0;
			first_position = number*mod;// n 
			last_position = non_zero - 1;//k 
			first_string = return_string(number*mod, row) - 1; //i 
			last_string = return_string(non_zero - 1, row) - 1;//j
			nsize = rest + first_string + size - 1 - last_string;

			val_ = new double[nsize]; // definition 
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					val_[i] = 0;
				}
				else
				{
					val_[i] = val[first_position + in1];
					in1++;
				}
			}
			//memcpy(&A_[first_string],&A[first_position],sizeof(double)*(rest));


			col_ = new int[nsize];
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					col_[i] = i;
				}
				else
				{
					col_[i] = col[first_position + in2];
					in2++;
				}
			}
			//memcpy(&B_[first_string], &B[first_position], sizeof(double)*(rest));

			row_ = new int[size + 1];

			for (int i = 0; i < first_string; i++) //0123..C..000 
				row_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				row_[count] = row[count] - first_position + first_string;
				if (row[count] - first_position < 0) row_[count] = first_string;
			}
			row_[size] = nsize;

		}
		else
		{
			int in1 = 0;
			int in2 = 0;
			first_position = number*mod;// n 
			last_position = (number + 1)*mod - 1;//k 
			first_string = return_string(number*mod, row) - 1; //i 
			last_string = return_string((number + 1)*mod - 1, row) - 1;//j 
			nsize = mod + first_string + size - 1 - last_string;

			val_ = new double[nsize]; // definition 
			for (int i = 0; i < nsize; i++)
			{
				if ((i < first_string) || (i > first_string + mod - 1))
				{
					val_[i] = 0;
				}
				else
				{
					val_[i] = val[first_position + in1];
					in1++;
				}
			}
			//memcpy(&A_[first_string], &A[first_position], sizeof(double)*(mod));

			col_ = new int[nsize];

			int inn = 1;
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					col_[i] = i;
				}
				else if (i < first_string + mod)
				{
					col_[i] = col[first_position + in2];
					in2++;
				}
				else
				{
					col_[i] = last_string + inn;
					inn++;
				}
			}
			//memcpy(&B_[first_string], &B[first_position], sizeof(double)*(mod));

			row_ = new int[size + 1];

			for (int i = 0; i < first_string; i++) //0123..C..000 
				row_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				row_[count] = row[count] - first_position + first_string;
				if (row[count] - first_position < 0) row_[count] = first_string;
			}
			int l = 1;
			for (int i = last_string + 1; i < size; i++) //0123..C..n.. 
			{
				row_[i] = first_string + last_position - first_position + l;
				l++;
			}
			row_[size] = nsize;

		}
#if CHECKER
		cout << endl << "Device: " << number << "  n:  " << first_position << "  k:  " << last_position << "  i:  " << first_string << "  j:  " << last_string << endl;
		cout << endl;
		for (int i = 0; i < nsize; i++)
		{
			cout << val_[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < nsize; i++)
		{
			cout << col_[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < size + 1; i++)
		{
			cout << row_[i] << " ";
		}
		cout << endl;
#endif
		temp[number] = nsize;

		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaMalloc((void **)&d_val[number], sizeof(double)*nsize));
		checkCudaErrors(cudaMalloc((void **)&d_col[number], sizeof(int)*nsize));
		checkCudaErrors(cudaMalloc((void **)&d_row[number], sizeof(int)*(size + 1)));
		checkCudaErrors(cudaMemcpy(d_val[number], val_, sizeof(double)*nsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_col[number], col_, sizeof(int)*nsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_row[number], row_, sizeof(int)*(size + 1), cudaMemcpyHostToDevice));

		delete[] val_;
		delete[] col_;
		delete[] row_;
	}
	return temp;
}

void GPU_mult(double* vec, int size, int* nnz, double* diag, int gpu_amount, double** d_val, int** d_col, int** d_row, double* rezult, int maximumThreads)
{
	double **pipe = new double*[gpu_amount];
	for (int i = 0; i < gpu_amount; i++)
	{
		pipe[i] = new double[size];
	}

	//size == vec.size()
	checkCudaErrors(cudaSetDevice(0));
	double *temp_rez;
	double *vec_temp;
	double * checking = new double[size];
	double** rez_p = new  double *[gpu_amount];
	//double** rez_h = new  double *[gpu_amount];

	checkCudaErrors(cudaMalloc((void**)&temp_rez,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&vec_temp, sizeof(double)*size));
	//checkCudaErrors(cudaMemset(temp_rez, 0.0, size));
	all_zero <<<10000, maximumThreads>>> (size,temp_rez);
	//Debuger(temp_rez, size, checking);
	//checkCudaErrors(cudaMemset(vec_temp, 0.0, size));
	all_zero << <10000, maximumThreads>> > (size, vec_temp);
	//Debuger(vec_temp, size, checking);
	double *one = new double;
	*one = 1.0;
	double *zero = new double;
	*zero = 0.0;
	//double *x_d;
	
	omp_set_num_threads(gpu_amount);
	double dtime = omp_get_wtime();
#pragma omp parallel for// private(rez_p)
	for (int number = 0; number < gpu_amount; number++)
	{
		cusparseHandle_t handle = NULL;
		cusparseMatDescr_t Adescr = NULL;
		checkCudaErrors(cudaSetDevice(number));
		//checkCudaErrors(cudaMalloc((void **)&x_d, sizeof(double)*size));
		//checkCudaErrors(cudaMalloc((void**)&tempnam,sizeof(double)*size));
		//checkCudaErrors(cudaMemcpy(x_d, vec, sizeof(double)*size, cudaMemcpyHostToDevice));
		checkCudaErrors(cusparseCreate(&handle));
		checkCudaErrors(cusparseCreateMatDescr(&Adescr));

		checkCudaErrors(cudaMalloc((void **)&rez_p[number], sizeof(double)*size));

		checkCudaErrors(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, nnz[number], one,
			Adescr,
			d_val[number],
			d_row[number], d_col[number],
			vec, zero,
			rez_p[number]));

		checkCudaErrors(cudaMemcpy(pipe[number], rez_p[number], sizeof(double)*size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(rez_p[number]));
		//checkCudaErrors(cudaFree(x_d));
		checkCudaErrors(cusparseDestroy(handle));
		checkCudaErrors(cusparseDestroyMatDescr(Adescr));
	}

		// trouble !!!

		checkCudaErrors(cudaSetDevice(0));
		for (int i = 0; i < gpu_amount; i++)
		{
			checkCudaErrors(cudaMemcpy(vec_temp, pipe[i], sizeof(double)*size, cudaMemcpyHostToDevice));
			//Debuger(vec_temp, size, checking);
			//Debuger(temp_rez, size, checking);
			vector_addition << <10000, maximumThreads >> > (vec_temp, size, temp_rez);
			//Debuger(temp_rez,size,checking);
			cudaDeviceSynchronize();
		}

	/*for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < gpu_amount; j++)
		{
			rezult[i] += pipe[j][i];
		}
	}*/
	
	//vector_addition << <10000, maximumThreads >> > (pipe,size,gpu_amount,rezult);

	/*for (int i = 0; i < size; i++)
	{
		rezult[i] += diag[i] * vec[i];
	}*/
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));
	checkCudaErrors(cudaSetDevice(0));
	not_full_scalar << <10000,maximumThreads >> > (diag,vec,size,vec_temp);
	//Debuger(vec_temp, size, checking);
	//cudaDeviceSynchronize();
	vector_addition << <10000, maximumThreads >> > (vec_temp, size, temp_rez);
	//Debuger(temp_rez, size, checking);
	//cudaDeviceSynchronize();
	checkCudaErrors(cublasDcopy(cublasHandle, size, temp_rez, 1, rezult, 1));

	checkCudaErrors(cudaFree(temp_rez));
//	Debuger(temp_rez, size, checking);
	checkCudaErrors(cudaFree(vec_temp));
	//Debuger(temp_rez, size, checking);
	//checkCudaErrors(cudaFree(rez_p));
	/*for (int i = 0; i < gpu_amount; i++)
	{
		delete pipe[i];
	}
	delete[] pipe;*/
//	delete vec_temp;
	delete zero;
	//delete x_d;
	delete one;
	//delete temp_rez;
	delete checking;
	delete[] pipe;
	delete[] rez_p;
}



void GPU_mult_for_little_gradient(double* vec, int size, int* nnz, double* diag, int gpu_amount, double** d_val, int** d_col, int** d_row, double* rezult)
{
	double **pipe = new double*[gpu_amount];
	for (int i = 0; i < gpu_amount; i++)
	{
	pipe[i] = new double[size];
	}
	//size == vec.size()
	//double *temp_rez;
	double** rez_p = new  double *[gpu_amount];
	//checkCudaErrors(cudaMallocManaged((void**)&temp_rez, sizeof(double)*size));
	double *one = new double;
	*one = 1.0;
	double *zero = new double;
	*zero = 0.0;
	double *x_d;
	omp_set_num_threads(gpu_amount);
	double dtime = omp_get_wtime();
#pragma omp parallel for// private(rez_p)
	for (int number = 0; number < gpu_amount; number++)
	{
		cusparseHandle_t handle = NULL;
		cusparseMatDescr_t Adescr = NULL;
		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaMalloc((void **)&x_d, sizeof(double)*size));
		//checkCudaErrors(cudaMalloc((void**)&tempnam,sizeof(double)*size));
		checkCudaErrors(cudaMemcpy(x_d, vec, sizeof(double)*size, cudaMemcpyHostToDevice));
		checkCudaErrors(cusparseCreate(&handle));
		checkCudaErrors(cusparseCreateMatDescr(&Adescr));

		checkCudaErrors(cudaMalloc((void **)&rez_p[number], sizeof(double)*size));

		checkCudaErrors(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, nnz[number], one,
			Adescr,
			d_val[number],
			d_row[number], d_col[number],
			x_d, zero,
			rez_p[number]));

		checkCudaErrors(cudaMemcpy(pipe[number], rez_p[number], sizeof(double)*size, cudaMemcpyDeviceToHost));

		//vector_addition << <10000, maximumThreads >> > (rez_p[number], size, number, temp_rez);
		checkCudaErrors(cudaFree(rez_p[number]));
		checkCudaErrors(cudaFree(x_d));
		checkCudaErrors(cusparseDestroy(handle));
		checkCudaErrors(cusparseDestroyMatDescr(Adescr));
	}

	for (int i = 0; i < size; i++)
	{
	for (int j = 0; j < gpu_amount; j++)
	{
	rezult[i] += pipe[j][i];
	}
	}

	//vector_addition << <10000, maximumThreads >> > (pipe,size,gpu_amount,rezult);

	for (int i = 0; i < size; i++)
	{
	rezult[i] += diag[i] * vec[i];
	}
	
	for (int i = 0; i < gpu_amount; i++)
	{
	delete pipe[i];
	}
	delete[] pipe;
	delete zero;
	delete one;
	delete[] rez_p;
}






double* gpu_solver::GPU_gradient_solver(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	int *temp = new int[gpu];

	double* r0 = new double[size];
	double* x0 = new double[size];
	double* x_k = new double[size];
	double* z0 = new double[size];
	double* z_k = new double[size];
	double* r_k = new double[size];
	double* ch = new double[size];
	double* cont = new double[size];
	double* testing = new double[size];

	//*r0 = right;// x0 ={0...}
	memcpy(r0, right, sizeof(double)*(size));
	memcpy(z0, r0, sizeof(double)*(size));
	memcpy(r_k, r0, sizeof(double)*(size));
	double a_k;
	double b_k;

	double r0_to_r0;
	double right_to_right = sqrt(dot_product(right, right, size));
	double rk_to_rk;

	double checking;
	bool fg = true;
	int step = 0;
	double gpu_time = 0;
	clock_t int1 = clock();

	for (int i = 0; i < size; i++)
	{
		x0[i] = 0;
	}
	temp = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);
	clock_t int2 = clock();
#if CHECKER
	cout << "SPLIT TIME:  " << double(int2 - int1) / 1000.0 << endl;
#endif
	do
	{
		if (!fg)
		{
			memcpy(r0, r_k, sizeof(double)*(size));
			memcpy(x0, x_k, sizeof(double)*(size));
			memcpy(z0, z_k, sizeof(double)*(size));
		}

		r0_to_r0 = dot_product(r0, r0, size);
		clock_t gpu_time1 = clock();
		memset(ch,0,sizeof(double)*(size));
		GPU_mult_for_little_gradient(z0, size, temp, diag, gpu, d_val, d_col, d_row, ch);
		clock_t gpu_time2 = clock();
		gpu_time += double(gpu_time2 - gpu_time1);
		a_k = r0_to_r0 / dot_product(ch, z0, size);

		vector_on_number(z0, a_k, size, cont);
		if (step == 640)
			cout << "640" << endl;
		sum_vector(x0, cont, size, x_k);

		vector_on_number(ch, a_k, size, cont);
		raznost_vector(r0, cont, size, r_k);

		rk_to_rk = dot_product(r_k, r_k, size);
		b_k = rk_to_rk / r0_to_r0;

		vector_on_number(z0, b_k, size, cont);
		sum_vector(r_k, cont, size, z_k);

		fg = false;
		step++;
		checking = sqrt(rk_to_rk) / right_to_right;
		

		//cout << endl<<"Checking" << checking << endl;
	} while ((checking >= APPROX) && (step < STEP_LIMIT));

	//cout <<endl<< "GPU TIME:  " << gpu_time / 1000.0 << endl;
	cout << "NEVAZKA:  " << checking << endl;
	GPU_mult_for_little_gradient(x_k, size, temp, diag, gpu, d_val, d_col, d_row, ch);
	raznost_vector(ch, right, size, testing);
	double verify = sqrt(dot_product(testing, testing, size));
	cout << endl << "VERIFICATION: " << verify << endl;
	cout << endl << "Step =	"  << step << endl;
    

	for (int number = 0; number < gpu; number++)
	{
		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaFree(d_val[number]));
		checkCudaErrors(cudaFree(d_col[number]));
		checkCudaErrors(cudaFree(d_row[number]));
	}
	
	delete[] temp;
	delete[] d_val;
	delete[] d_col;
	delete[] d_row;
	delete[] ch;
	delete[] cont;
	delete[] x0;
	delete[] r0;
	//delete[] z0;
	//delete[] z_k;
	//delete[] r_k;
	//delete[] testing;
	return x_k;
}

double* gpu_solver::GPU_stab_bi_gradient_solver(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));
	//double *test = new double[size];
	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];
	
	//Array with devicearray's sizes
	int *temp = new int[gpu];
	temp = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);
	//int step = 0;
	bool flag = true;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;
	//Initialization of diag
	double* final_result = new double[size];
	//Initialization of all variables
	checkCudaErrors(cudaSetDevice(0));
	cublasHandle_t cublasHandle = NULL;
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));
	checkCudaErrors(cublasCreate(&cublasHandle));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0.0, sizeof(double)*(size)));
	double *r0, *rT;
	double * diag_gpu;
	double * h;
	//double * right_part_gpu;
	checkCudaErrors(cudaMalloc((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rT, sizeof(double)*(size)));
	//checkCudaErrors(cudaMalloc((void **)&right_part_gpu, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&diag_gpu, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&h, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(diag_gpu, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rT, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(right_part_gpu,right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	double *rho0 = new double;
	double *alpha0 = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha0 = 1.0;	
	*omega0 = 1.0;
	double *minus_one = new double;
	*minus_one = -1.0;
	double *nu0, *p0;
	int step = 0;
	checkCudaErrors(cudaMalloc((void **)&nu0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&p0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(nu0, 0.0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(p0, 0.0, sizeof(double)*(size)));
	double *rhoK = new double;
	double *alphaK = new double;
	double *omegaK = new double;
	double *betaK = new double;
	double *pK, *nuK, *sK, *tK, *xK, *rK;
	checkCudaErrors(cudaMalloc((void **)&pK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&nuK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&sK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&tK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	double *temp1, *temp2, *temp3;
	checkCudaErrors(cudaMalloc((void **)&temp1, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&temp2, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&temp3, sizeof(double)*(size)));
	//double * NegOmega=new double;
	double * temp_var_1=new double;
	double * temp_var_2 = new double;
	double * checking=new double[size];
	//*NegOmega = -(*omega0);
	//1
	do
	{
		cublasDdot(cublasHandle, size, r0, 1, rT, 1, rhoK);
		//2
		*betaK = (*rhoK / *rho0) * (*alpha0 / *omega0);
		//cout <<"OUT: "<< *betaK;
		//3
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (nu0, -(*omega0), size, temp1);


		cublasDaxpy(cublasHandle, size, one, p0, 1, temp1, 1);
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (temp1, *betaK, size, temp1);
		cublasDaxpy(cublasHandle, size, one, rT, 1, temp1, 1);
		cublasDcopy(cublasHandle, size, temp1, 1, pK, 1);
		Debuger(pK, size,checking);
		//4


		GPU_mult(pK, size, temp, diag_gpu, gpu, d_val, d_col, d_row, nuK, deviceProp.maxThreadsPerBlock);
		Debuger(nuK, size,checking);
		//5


		cublasDdot(cublasHandle, size, r0, 1, nuK, 1, temp_var_1);
		
		*alphaK = (*rhoK) / (*temp_var_1);
		
		//6
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (pK, *alphaK, size, temp1);
		cudaDeviceSynchronize();
		
		cublasDaxpy(cublasHandle, size, one, x0, 1, temp1, 1);
		cublasDcopy(cublasHandle, size, temp1, 1, h, 1);
		Debuger(h, size,checking);
		//7
		

		//cublasDaxpy(cublasHandle, size, minus_one, h, 1, xK, 1);
		

		//8
		Debuger(nuK,size,checking);
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (nuK, -(*alphaK), size, temp1);
		cudaDeviceSynchronize();
		Debuger(rT, size, checking);
		Debuger(temp1, size, checking);
		cublasDaxpy(cublasHandle, size, one, rT, 1, temp1, 1);
		Debuger(temp1, size, checking);
		cublasDcopy(cublasHandle, size, temp1, 1, sK, 1);
		Debuger(sK, size,checking);
		
		//9
		GPU_mult(sK, size, temp, diag_gpu, gpu, d_val, d_col, d_row, tK, deviceProp.maxThreadsPerBlock);
		//Debuger(xK, size,checking);
		Debuger(tK,size,checking);
		//10
		cublasDdot(cublasHandle, size, tK, 1, sK, 1, temp_var_1);
		//Debuger(temp_var_1,size,checking);
		cublasDdot(cublasHandle, size, tK, 1, tK, 1, temp_var_2);
		//Debuger(temp_var_2, size, checking);
		*omegaK = *temp_var_1 / *temp_var_2;
		//11
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (sK, *omegaK, size, temp1);
		cudaDeviceSynchronize();
		cublasDaxpy(cublasHandle, size, one, h, 1, temp1, 1);	
		cublasDcopy(cublasHandle, size, temp1, 1, xK, 1);
		Debuger(xK,size,checking);
		//12
		/*cublasDaxpy(cublasHandle, size, minus_one, xK, 1, x0, 1);
		cublasDnrm2(cublasHandle, size, x0, 1, temp_var_1);*/
		
		//13
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (tK, -(*omegaK), size, temp1);
		
		cublasDaxpy(cublasHandle, size, one, sK, 1, temp1, 1);
		cublasDcopy(cublasHandle, size, temp1, 1, rK, 1);
		Debuger(rK, size, checking);
		cublasDnrm2(cublasHandle, size, rK, 1, temp_var_1);
		cublasDnrm2(cublasHandle, size, r0,1,temp_var_2);
		//if(step%20==0)
		//cout <<"NEVAZKA = "<< *temp_var_1/ *temp_var_2 << endl;
		if (*temp_var_1 / *temp_var_2<= APPROX)
		{
			cout <<endl<< "NEVAZKA = " << *temp_var_1/ *temp_var_2 << endl;
			checkCudaErrors(cudaMemcpy(final_result,xK, sizeof(double)*size, cudaMemcpyDefault));
			break;
		}


		cublasDcopy(cublasHandle, size, rK, 1, rT, 1);
		cublasDcopy(cublasHandle, size, xK, 1, x0, 1);
		cublasDcopy(cublasHandle, size, pK, 1, p0, 1);
		cublasDcopy(cublasHandle, size, nuK, 1, nu0, 1);
		*rho0 = *rhoK;
		*omega0 = *omegaK;
		*alpha0 = *alphaK;

		step++;
		//cout <<"Step = "<< step << endl;
	} while (step<=STEP_LIMIT);
	//Verification
	GPU_mult(xK, size, temp, diag_gpu, gpu, d_val, d_col, d_row, temp1, deviceProp.maxThreadsPerBlock);
	cublasDaxpy(cublasHandle,size,minus_one,r0,1,temp1,1);
	cublasDnrm2(cublasHandle, size, temp1, 1, temp_var_1);
	cout <<endl<< "VERIFICATION:  " << *temp_var_1 << endl;
	checkCudaErrors(cudaFree(r0));
	checkCudaErrors(cudaFree(rK));
	checkCudaErrors(cudaFree(x0));
	checkCudaErrors(cudaFree(xK));
	checkCudaErrors(cudaFree(pK));
	checkCudaErrors(cudaFree(p0));
	//checkCudaErrors(cudaFree(right_part_gpu));
	checkCudaErrors(cudaFree(nuK));
	checkCudaErrors(cudaFree(nu0));                 
	checkCudaErrors(cudaFree(temp1));
	checkCudaErrors(cudaFree(temp2));
	checkCudaErrors(cudaFree(temp3));
	checkCudaErrors(cudaFree(sK));
	checkCudaErrors(cudaFree(tK));
	checkCudaErrors(cudaFree(h));
	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	cout <<endl<< "STEPS: = " << step << endl;
	delete temp_var_1;
	delete temp_var_2;


	return final_result;

	
}






double* gpu_solver::GPU_stab_bi_gradient_solver_with_preconditioner(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{

	

	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));
	double *test = new double[size];
	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	//Array with devicearray's sizes
	int *temp = new int[gpu];
	temp = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);
	//int step = 0;
	bool flag = true;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;
	//Initialization of diag
	double* final_result = new double[size];
	//Initialization of all variables
	checkCudaErrors(cudaSetDevice(0));
	cublasHandle_t cublasHandle = NULL;
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));
	checkCudaErrors(cublasCreate(&cublasHandle));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0, sizeof(double)*(size)));
	double *r0, *rT;
	double * diag_gpu;
	double * h;
	
	checkCudaErrors(cudaMalloc((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rT, sizeof(double)*(size)));
	//checkCudaErrors(cudaMalloc((void **)&right_part_gpu, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&diag_gpu, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&h, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(diag_gpu, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rT, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	double *rho0 = new double;
	double *alpha0 = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha0 = 1.0;
	*omega0 = 1.0;
	double *minus_one = new double;
	*minus_one = -1.0;
	double *nu0, *p0;
	int step = 0;
	checkCudaErrors(cudaMalloc((void **)&nu0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&p0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(nu0, 0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(p0, 0, sizeof(double)*(size)));
	double *rhoK = new double;
	double *alphaK = new double;
	double *omegaK = new double;
	double *betaK = new double;
	double *pK, *nuK, *sK, *tK, *xK, *rK;
	checkCudaErrors(cudaMalloc((void **)&pK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&nuK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&sK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&tK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	double *temp1, *temp2, *temp3,*temp4,*temp5;
	checkCudaErrors(cudaMalloc((void **)&temp1, sizeof(double)*(size)));   // for many
	checkCudaErrors(cudaMalloc((void **)&temp2, sizeof(double)*(size)));	//K^(-1)
	checkCudaErrors(cudaMalloc((void **)&temp3, sizeof(double)*(size)));	//z
	checkCudaErrors(cudaMalloc((void **)&temp4, sizeof(double)*(size)));		//ddot tk*K^-1
	checkCudaErrors(cudaMalloc((void **)&temp5, sizeof(double)*(size)));		//ddot sk*^(-1)
	
	double * temp_var_1 = new double;
	double * temp_var_2 = new double;
	diag_revers << <10000, deviceProp.maxThreadsPerBlock >> > (diag_gpu, temp2, size);
	
	double *checking = new double[size];
	do
	{
		//1
		cublasDdot(cublasHandle, size, r0, 1, rT, 1, rhoK);
		//2
		*betaK = (*rhoK / *rho0) * (*alpha0 / *omega0);
		//cout <<"OUT: "<< *betaK;
		//3
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (nu0, -(*omega0), size, temp1);
		Debuger(temp1, size, checking);
		cublasDaxpy(cublasHandle, size, one, p0, 1, temp1, 1);
		Debuger(temp1, size, checking);
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (temp1, *betaK, size, temp1);
		cublasDaxpy(cublasHandle, size, one, rT, 1, temp1, 1);
		cublasDcopy(cublasHandle, size, temp1, 1, pK, 1);
		Debuger(pK, size, checking);
		//4

		
		not_full_scalar << <10000, deviceProp.maxThreadsPerBlock >> > (temp2,pK,size,temp3);
		//Debuger(temp3, size,checking);
		Debuger(temp3, size, checking);
		//5

		GPU_mult(temp3, size, temp, diag_gpu, gpu, d_val, d_col, d_row, nuK, deviceProp.maxThreadsPerBlock);
		Debuger(nuK, size,checking);
		Debuger(nuK, size, checking);
		//6


		cublasDdot(cublasHandle, size, r0, 1, nuK, 1, temp_var_1);

		*alphaK = (*rhoK) / (*temp_var_1);

		//7
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (temp3, *alphaK, size, temp1);
		Debuger(temp1, size, checking);
		cublasDaxpy(cublasHandle, size, one, x0, 1, temp1, 1);
		cublasDcopy(cublasHandle, size, temp1, 1, h, 1);
		Debuger(h, size, checking);
		//7


		//cublasDaxpy(cublasHandle, size, minus_one, h, 1, xK, 1);


		//9
		//Debuger(temp1, size);
		//Debuger(temp1, size, checking);
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (nuK, -(*alphaK), size, temp1);
		//Debuger(temp1, size);
		Debuger(temp1, size, checking);
		Debuger(rT, size, checking);
		cublasDaxpy(cublasHandle, size, one, rT, 1, temp1, 1);
		//Debuger(temp1, size);
		//Debuger(sK, size, checking);
		Debuger(temp1,size,checking);
		cublasDcopy(cublasHandle, size, temp1, 1, sK, 1);
		Debuger(sK, size,checking);
		//10
		Debuger(sK, size,checking);
		not_full_scalar << <10000, deviceProp.maxThreadsPerBlock >> > (temp2, sK, size, temp3);
		Debuger(temp3, size,checking);
		//11
		GPU_mult(temp3, size, temp, diag_gpu, gpu, d_val, d_col, d_row, tK, deviceProp.maxThreadsPerBlock);
		Debuger(tK, size,checking);
		
		//12
		not_full_scalar << <10000, deviceProp.maxThreadsPerBlock >> > (temp2, tK, size, temp4);
		not_full_scalar << <10000, deviceProp.maxThreadsPerBlock >> > (temp2, sK, size, temp5);
		Debuger(temp4, size, checking);
		Debuger(temp5, size, checking);
		cublasDdot(cublasHandle, size, temp4, 1, temp5, 1, temp_var_1);
		cublasDdot(cublasHandle, size, temp4, 1, temp4, 1, temp_var_2);
		*omegaK = *temp_var_1 / *temp_var_2;
		//13
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock>> > (temp3, *omegaK, size, temp1);
		Debuger(temp1, size, checking);
		cublasDaxpy(cublasHandle, size, one, h, 1, temp1, 1);
		Debuger(temp1, size, checking);
		cublasDcopy(cublasHandle, size, temp1, 1, xK, 1);
 		Debuger(xK, size,checking);
		//12
		/*cublasDaxpy(cublasHandle, size, minus_one, xK, 1, x0, 1);
		cublasDnrm2(cublasHandle, size, x0, 1, temp_var_1);*/

		//15
		vec_mul_number << <10000, deviceProp.maxThreadsPerBlock>> > (tK, -(*omegaK), size, temp1);
		Debuger(temp1, size, checking);
		cublasDaxpy(cublasHandle, size, one, sK, 1, temp1, 1);
		Debuger(temp1, size,checking);
		cublasDcopy(cublasHandle, size, temp1, 1, rK, 1);
 		Debuger(rK, size,checking);
		cublasDnrm2(cublasHandle, size, rK, 1, temp_var_1);
		cublasDnrm2(cublasHandle, size, r0, 1, temp_var_2);
		//if(step%20==0)
		//cout <<"NEVAZKA = "<< *temp_var_1 / *temp_var_2 << endl;
		//if (*temp_var_1/ *temp_var_2 <= APPROX || *omegaK<=APPROX)
		if (*temp_var_1 / *temp_var_2 <= APPROX )
		{
			//cout << endl << "NEVAZKA = " << *temp_var_1/ *temp_var_2<< endl;
			checkCudaErrors(cudaMemcpy(final_result, xK, sizeof(double)*size, cudaMemcpyDefault));
			break;
		}


		cublasDcopy(cublasHandle, size, rK, 1, rT, 1);
		cublasDcopy(cublasHandle, size, xK, 1, x0, 1);
		cublasDcopy(cublasHandle, size, pK, 1, p0, 1);
		cublasDcopy(cublasHandle, size, nuK, 1, nu0, 1);
		*rho0 = *rhoK;
		*omega0 = *omegaK;
		*alpha0 = *alphaK;

		step++;
		//cout <<"Step = "<< step << endl;
	} while (step <= STEP_LIMIT);
	//Verification
	GPU_mult(xK, size, temp, diag_gpu, gpu, d_val, d_col, d_row, temp1, deviceProp.maxThreadsPerBlock);
	cublasDaxpy(cublasHandle, size,minus_one, r0, 1, temp1, 1);
	cublasDnrm2(cublasHandle, size, temp1, 1, temp_var_1);
	//cout << endl << "VERIFICATION:  " << *temp_var_1 << endl;
	checkCudaErrors(cudaFree(r0));
	checkCudaErrors(cudaFree(rK));
	checkCudaErrors(cudaFree(x0));
	checkCudaErrors(cudaFree(xK));
	checkCudaErrors(cudaFree(pK));
	checkCudaErrors(cudaFree(p0));
	//checkCudaErrors(cudaFree(right_part_gpu));
	checkCudaErrors(cudaFree(nuK));
	checkCudaErrors(cudaFree(nu0));
	checkCudaErrors(cudaFree(temp1));
	checkCudaErrors(cudaFree(temp2));
	checkCudaErrors(cudaFree(temp3));
	checkCudaErrors(cudaFree(sK));
	checkCudaErrors(cudaFree(temp4));
	checkCudaErrors(cudaFree(temp5));
	checkCudaErrors(cudaFree(tK));
	checkCudaErrors(cudaFree(h));
	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	//cout << endl << "STEPS: = " << step << endl;
	delete temp_var_1;
	delete temp_var_2;


	return final_result;


}



 void gpu_solver::matrix_eigenvalues(double *val, int *col, int *row, double *diag, int non_zero, int size, double * eigen_values, double** eigen_vectors,double * b, double *val2, int *col2, int *row2, double *diag2, int non_zero2, int size2, int amount_ev)
{
	 

	 int gpu;
	 checkCudaErrors(cudaGetDeviceCount(&gpu));
	 double ** d_val = new  double *[gpu];
	 int ** d_col = new int *[gpu];
	 int ** d_row = new int *[gpu];

	 double ** d_val2 = new  double *[gpu];
	 int ** d_col2 = new int *[gpu];
	 int ** d_row2 = new int *[gpu];

	 //Array with devicearray's sizes
	 int *temp = new int[gpu];
	 int *temp_M = new int[gpu];
	 temp = split(gpu, val, col, row, size, non_zero, d_val, d_col, d_row);
	 temp_M = split(gpu, val2, col2, row2, size2, non_zero2, d_val2, d_col2, d_row2);
	 checkCudaErrors(cudaSetDevice(0));
	 cublasHandle_t cublasHandle = NULL;
	 cusparseHandle_t cusparseHandle = NULL;
	 checkCudaErrors(cusparseCreate(&cusparseHandle));
	 cusparseMatDescr_t matDescr = NULL;
	 checkCudaErrors(cusparseCreateMatDescr(&matDescr));
	 checkCudaErrors(cublasCreate(&cublasHandle));
	 cudaDeviceProp deviceProp;
	 double *diag_gpu;
	 checkCudaErrors(cudaMalloc((void **)&diag_gpu, sizeof(double)*(size)));
	 checkCudaErrors(cudaMemcpy(diag_gpu, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));
	 double *diag_gpu2;
	 checkCudaErrors(cudaMalloc((void **)&diag_gpu2, sizeof(double)*(size2)));
	 checkCudaErrors(cudaMemcpy(diag_gpu2, diag2, sizeof(double)*(size2), cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

	 double* nu_vec,
	 double * alpha_vec;
	 double * beta_vec;
	 double* w_vec ;       // myvector
	 double* alpha_j=new double;
	 double* t=new double;
	 double * beta_new;
	 double * checking = new double[size];
	 double * matrix_w;
	 double * multi_temp;
	 double * matr_dense;
	 double * eigen_values_gpu;
	 double * eigenvectors_gpu;
	 int * CONVERGE_AMOUNT_CPU = new int;
	 *CONVERGE_AMOUNT_CPU = 0;
	 int * CONVERGE_AMOUNT;
	 checkCudaErrors(cudaMalloc((void **)&CONVERGE_AMOUNT, sizeof(int)));
	 checkCudaErrors(cudaMemset(CONVERGE_AMOUNT, 0, sizeof(int)));
	 //*CONVERGE_AMOUNT = 0;    //current converge
	 double * converge_eig_vec;   //result converge vectors
	 double * converge_eig_val;    //result converge values
	 double *converge_eig_val_numb_T;   //temp for ind of converge values in array
	 double *converge_temp ;      
	 //*converge_temp = 0;
	 checkCudaErrors(cudaMalloc((void **)&converge_temp, sizeof(double)));
	 checkCudaErrors(cudaMemset(converge_temp, 0.0, sizeof(double)));
	 double* temp_ev; //temp
	 checkCudaErrors(cudaMalloc((void **)&temp_ev, sizeof(double)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&converge_eig_val_numb_T, sizeof(double)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&converge_eig_vec, sizeof(double)*(size)*size));
	 checkCudaErrors(cudaMalloc((void **)&converge_eig_val, sizeof(double)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&eigenvectors_gpu, sizeof(double)*(amount_ev)*(size)));
	 checkCudaErrors(cudaMalloc((void **)&eigen_values_gpu, sizeof(double)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&matr_dense, sizeof(double)*(amount_ev)*amount_ev));
	 checkCudaErrors(cudaMemset(matr_dense, 0.0, sizeof(double)*(amount_ev)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&matrix_w, sizeof(double)*(size)*(amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&multi_temp, sizeof(double)*(size)*(amount_ev)));
	 checkCudaErrors(cudaMemset(matrix_w, 0.0, sizeof(double)*(size*amount_ev)));
	 checkCudaErrors(cudaMalloc((void **)&nu_vec, sizeof(double)*(size)));
	 checkCudaErrors(cudaSetDevice(0));
	 checkCudaErrors(cudaMemset(nu_vec, 0, sizeof(double)*(size)));
	 checkCudaErrors(cudaMalloc((void **)&alpha_vec, sizeof(double)*(size)));
	 checkCudaErrors(cudaMalloc((void **)&beta_new, sizeof(double)));
	 checkCudaErrors(cudaMemset(alpha_vec, 0, sizeof(double)*(size)));
	 checkCudaErrors(cudaMemset(beta_new, 0.0, sizeof(double)));
	 checkCudaErrors(cudaMalloc((void **)&beta_vec, sizeof(double)*(amount_ev-1)));
	 checkCudaErrors(cudaMemset(beta_vec, 0.0, sizeof(double)*(amount_ev-1)));
	 checkCudaErrors(cudaMalloc((void **)&w_vec, sizeof(double)*(size)));                     // myvector
	 double *right_gpu;
	 double *  right_gpu_input_once;
	 checkCudaErrors(cudaMalloc((void **)&right_gpu_input_once, sizeof(double)*(size)));
	 checkCudaErrors(cudaMemcpy(right_gpu_input_once, b, sizeof(double)*(size), cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMalloc((void **)&right_gpu, sizeof(double)*(size)));
	 checkCudaErrors(cudaMemcpy(right_gpu, b, sizeof(double)*(size), cudaMemcpyHostToDevice));
	 //checkCudaErrors(cudaMemcpy(w_vec, b, sizeof(double)*(size), cudaMemcpyHostToDevice));
	 double *temp1_b;
	 checkCudaErrors(cudaMalloc((void **)&temp1_b, sizeof(double)*(size)));
	 //checkCudaErrors(cudaMemset(w_vec, 2.0, sizeof(double)*(size)));
	// matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, 0, w_vec, size);
	 double *tmp5;
	 checkCudaErrors(cudaMalloc((void **)&tmp5, sizeof(double)));
	 double *tmp6;
	 checkCudaErrors(cudaMalloc((void **)&tmp6, sizeof(double)));
	 double * temp2;
	 checkCudaErrors(cudaMalloc((void **)&temp2, sizeof(double)*(size)));
	 double * temp3;
	 checkCudaErrors(cudaMalloc((void **)&temp3, sizeof(double)*(size)));
	 double * temp6;
	 checkCudaErrors(cudaMalloc((void **)&temp6, sizeof(double)*(size)));
	 double *tmp1=new double;
	 double *tmp_2 = new double;
	 double *tmp3 = new double;
	 int* tmp_int = new int;
	 bool main_flag = false;
	 double * zero_f = new double;
	 *zero_f = 0;
	 double*tmp2 = new double;
	 double * one = new double;
	 *one = 1;
	 double *b_i = new double;
	 double * b_i_old = new double;
	 *b_i_old = 0;
	 bool exit = false;
	 double *x_temp = new double[size];
	 double * temp1;   //right_gpu /aka first vector
	 checkCudaErrors(cudaMalloc((void **)&temp1, sizeof(double)*(size)));
	 double *x_temp_tld = new double[size];			//x^~
	 int step = 0;
	 double * last_beta = new double;
	 bool first_flag = false;
	 checkCudaErrors(cudaMalloc((void **)&x_temp, sizeof(double)*(size)));
	 checkCudaErrors(cudaMalloc((void **)&x_temp_tld, sizeof(double)*(size)));

	 double * minus_one = new double;
	 *minus_one = -1.0;
	 while (*CONVERGE_AMOUNT_CPU != amount_ev)
	 {
		 cout << endl << "THE " << step << " STAGE IS RINNING!" << endl;
		 //checkCudaErrors(cudaDeviceSynchronize());
		 if (first_flag != false)
		 {
			 bool flag_cor;
				 //Gramma-Shmidt procedure
				 for (int j = 0; j < *CONVERGE_AMOUNT_CPU; j++)
				 {
					 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_vec, j, size, temp1);
					 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, right_gpu_input_once, 1, tmp1));
					 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, temp1, 1, tmp2));
					 *tmp1 = (-1)*(*tmp1) / (*tmp2);
					 checkCudaErrors(cublasDaxpy(cublasHandle, size, tmp1, temp1, 1, right_gpu, 1));
					 Debuger(temp1,size,checking);
                 }
				
				 flag_cor = true;
				 for (int j = 0; j < *CONVERGE_AMOUNT_CPU; j++)
				 {
					 
					 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_vec, j, size, temp1);
					 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, right_gpu, 1, tmp1));
					 if (abs(*tmp1) > SK_Nev)
					 {
						 flag_cor = false;
					 }
					//cout << "Checking:  " << abs(*tmp1) << endl;
				 }
				 if (flag_cor == false)
				 {
					 cout <<endl<< "BAD G_SH" << endl;
				 }
				 else { cout<<endl << "CORRECT G_SH" << endl; }

				 

			// } while (flag_cor != true);
			// matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, 0, right_gpu, size);

		 }
		// all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (size, temp1);
		/* Debuger(right_gpu, size, checking);
		 Debuger(temp1, size, checking);*/
		 GPU_mult(right_gpu, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1, deviceProp.maxThreadsPerBlock);
		 cublasDdot(cublasHandle, size, temp1, 1, right_gpu, 1, tmp1);
         *tmp_2 = sqrt(*tmp1);
		 Debuger(temp1, size, checking);
		 vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (right_gpu, 1 / (*tmp_2), size, w_vec);		//first vector of Q matrix
		 Debuger(w_vec, size, checking);
		 matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, 0, w_vec, size);
		 Debuger(matrix_w, size, checking);
		 
				//x^-
		


		 double * proc_temp = new double[size];
		 *one = 1;
		 bool flag_cor = true;
		 
		 for (int i = 0; i < amount_ev; i++)   //этап ланцоша
		 {
			// if(exit==false)
			 
			 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, i, size, w_vec);
			 Debuger(w_vec, size, checking);
			 GPU_mult(w_vec, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1, deviceProp.maxThreadsPerBlock); // M_xi
			 matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (multi_temp, i, temp1, size);
			 Debuger(temp1, size, checking);
			 checkCudaErrors(cudaMemcpy(proc_temp, temp1, sizeof(double)*size, cudaMemcpyDeviceToHost));
			 proc_temp = GPU_stab_bi_gradient_solver_with_preconditioner(val, col, row, proc_temp, diag, non_zero, size);  // x^-
			 checkCudaErrors(cudaMemcpy(x_temp, proc_temp, sizeof(double)*(size), cudaMemcpyHostToDevice));
			 cublasDdot(cublasHandle, size, temp1, 1, x_temp, 1, tmp1);	//alpha
			 element << <1, 1 >> > (alpha_vec, i, *tmp1);//alpha_i
			// if (i != amount_ev - 1)
			 
				 double * checking1 = new double[size*(i + 1)];
				 *tmp1 = -*tmp1; //-alpha_i

				 Debuger(temp1, size, checking);
				 vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (w_vec, (*tmp1), size, temp1);
				 checkCudaErrors(cublasDaxpy(cublasHandle, size, one, temp1, 1, x_temp, 1));//end
				 if (i != 0)
				 {
					 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, i - 1, size, temp2);
					 // (*b_i_old) = -(*b_i_old);
					 vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (temp2, -(*b_i_old), size, temp2);  //
					 Debuger(temp2, size, checking);
					 checkCudaErrors(cublasDaxpy(cublasHandle, size, one, temp2, 1, x_temp, 1));  //x_temp = x^~i  
					 copy_v << <1000, deviceProp.maxThreadsPerBlock >> > (x_temp, size, temp6);
					 //Debuger(temp2, size, checking);
					 for (int s = 0; s < i; s++)    //full ortagonalization G_SH
					 {
						 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (multi_temp, s, size, temp3);
						 checkCudaErrors(cublasDdot(cublasHandle, size, temp3, 1, x_temp, 1, tmp3));
						 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, s, size, temp3);
						 (*tmp3) = -*tmp3;
						 vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (temp3, *tmp3, size, temp3);
						 checkCudaErrors(cublasDaxpy(cublasHandle, size, one, temp3, 1, x_temp, 1));
					 }

					 for (int j = 0; j < *CONVERGE_AMOUNT_CPU; j++)
					 {
						 //temp3
						 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_vec, j, size, temp3);
						 GPU_mult(temp3, size, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp2, deviceProp.maxThreadsPerBlock);
						 checkCudaErrors(cublasDdot(cublasHandle, size, temp6, 1, temp2, 1, tmp1));
						 *tmp1 = -*tmp1;
						 checkCudaErrors(cublasDaxpy(cublasHandle, size, tmp1, temp2, 1, x_temp, 1));

					 }

				 }

				 GPU_mult(x_temp, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1, deviceProp.maxThreadsPerBlock);
				 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, x_temp, 1, tmp1));
				 Debuger(temp1, size, checking);
				 Debuger(x_temp, size, checking);


				 if (i != amount_ev - 1)
				 {
					 *b_i_old = sqrt(*tmp1);
					 element << <1, 1 >> > (beta_vec, i, *b_i_old);
					 //Debuger(x_temp, size, checking);
					 vec_mul_number << <10000, deviceProp.maxThreadsPerBlock >> > (x_temp, 1 / (*b_i_old), size, x_temp); //
					 Debuger(x_temp, size, checking);



					 flag_cor = true;
					 for (int j = 0; j < i /*+ 1*/; j++)
					 {
						 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (multi_temp, j, size, temp1);
						 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, x_temp, 1, tmp1));
						 if (abs(*tmp1) > SK_Nev)
						 {
							 flag_cor = false;
							 cout << endl << "FUck in Lanc " << j;
						 }
						 // cout << "Checking:  " << abs(*tmp1) << endl;
					 }
					 for (int j = 0; j < *CONVERGE_AMOUNT_CPU /*+ 1*/; j++)
					 {
						 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_val, j, size, temp1);
						 GPU_mult(temp1, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1, deviceProp.maxThreadsPerBlock);
						 checkCudaErrors(cublasDdot(cublasHandle, size, temp1, 1, x_temp, 1, tmp1));
						 if (abs(*tmp1) > SK_Nev)
						 {
							 flag_cor = false;
							 cout << endl << "FUck in eigenvec " << j;
						 }
						 // cout << "Checking:  " << abs(*tmp1) << endl;
					 }
					 if (flag_cor == true)
						 cout <<endl<< "CHECKING COMPLETED AT STAGE " << i << endl;
					 else
					 {
						 cout <<endl<< "CHECKING is not COMPLETED AT STAGE " << i << endl;
						// exit = true;
					 }

					// if (exit == false)
						 matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (matrix_w, i + 1, x_temp, size);
				 }


				 else {
					 *last_beta = sqrt(*tmp1);
				 } 
		 }
			 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (amount_ev*amount_ev, matr_dense);
			 cusolverDnHandle_t cusolverH = NULL;
			 int info_gpu = 0;
			 checkCudaErrors(cusolverDnCreate(&cusolverH));
			 int  lwork = 0;
			 const int lda = amount_ev /*i + 1*/;
			 connect_diag_matr << <10000, deviceProp.maxThreadsPerBlock >> > (matr_dense, alpha_vec, beta_vec,amount_ev);
			 double* checking_mat = new double[(amount_ev*amount_ev)];
			// cout << "MATRIX T: " << endl;
			// Debuger_for_matr(matr_dense, amount_ev, amount_ev, checking_mat);
			// cout << endl;
			 cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
			 cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
			 checkCudaErrors(cusolverDnDsyevd_bufferSize(					//allocated memory for buffer
				 cusolverH,
				 jobz,
				 uplo,
				amount_ev, /*i + 1,*/
				 matr_dense,
				 lda,
				 eigen_values_gpu,
				 &lwork
			 ));

			 int *devInfo = NULL;
			 checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));
			 double *d_work = NULL;
			 checkCudaErrors(cudaMalloc((void**)&d_work, sizeof(double)*lwork));
			 checkCudaErrors(cusolverDnDsyevd(					//solver
				 cusolverH,
				 jobz,
				 uplo,
				amount_ev,// i + 1,
				 matr_dense,        //eigenvectors for T
				 lda,
				 eigen_values_gpu,   //eigenvalues for T
				 d_work,
				 lwork,
				 devInfo
			 ));

			 checkCudaErrors(cudaDeviceSynchronize());
			

		   ret_val << <1, 1 >> > (beta_vec,amount_ev-2,tmp5);
		   double * a = new double[amount_ev];
		   Debuger(beta_vec,amount_ev-1,a);
		   double * ch3 = new double;
		   double* checking_matr = new double[(amount_ev*amount_ev)];
		   cout << endl << "TEMPORARY EIGEN VALUES:" << endl;
		   double*ch_v = new double[amount_ev];
		   Debuger(eigen_values_gpu, amount_ev, ch_v);
		   for (int i = 0; i < amount_ev; i++)
			   cout << "Value: " << 1/ch_v[i] << endl;
		   cout << endl << "TEMPORARY EIGEN VECTOR:" << endl;
		   


		   Debuger_for_matr(matr_dense, amount_ev, amount_ev, checking_matr);
		   Debuger(tmp5, 1, ch3); 
		   cout << endl<<"BETA_q=" << *last_beta << endl;
		   element << <1, 1 >> > (tmp5, 0, *last_beta);
		   proverb << <1, 1 >> > (matr_dense,tmp5,amount_ev, eps_for_b,CONVERGE_AMOUNT, converge_eig_val_numb_T, converge_temp); 
		   //converge_eig_val_numb_T - array of places of converged current values in eigen_values_gpu
		   //converge_temp - number of converged values on  current stage

		 //checkCudaErrors(cudaDeviceSynchronize());
		   checkCudaErrors(cudaMemcpy(tmp3, converge_temp, sizeof(double), cudaMemcpyDeviceToHost));
		   checkCudaErrors(cudaMemcpy(CONVERGE_AMOUNT_CPU, CONVERGE_AMOUNT, sizeof(int), cudaMemcpyDeviceToHost));
		   cout <<endl<<"Converged values "<< *tmp3 << endl;
		 for (int i = 0; i < *tmp3; i++)
		 {
			 add_to_converge_values << <1, 1 >> > (converge_eig_val_numb_T, i, eigen_values_gpu,converge_eig_val, CONVERGE_AMOUNT,tmp5);
			 checkCudaErrors(cudaMemcpy(tmp1, tmp5, sizeof(double), cudaMemcpyDeviceToHost));
			 /*ret_val << <1, 1 >> > (converge_temp,i,tmp5);
			 checkCudaErrors(cudaMemcpy(tmp1, tmp5, sizeof(double), cudaMemcpyDeviceToHost));
			 ret_val << <1, 1 >> > (eigen_values_gpu,int(*tmp1),tmp6);
			 element << <1, 1 >> > (converge_eig_val, *CONVERGE_AMOUNT, *tmp6);*/
			 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (matr_dense, int(*tmp1), amount_ev, temp_ev);
			 checkCudaErrors(cublasDgemv(cublasHandle,
				 CUBLAS_OP_N,
				 size,
				 amount_ev,
				 one,
				 matrix_w,
				 size,
				 temp_ev,
				 1,
				 zero_f,
				 temp3,
				 1));
			 matr_add << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_vec, *CONVERGE_AMOUNT_CPU+i, temp3, size);
			// first_flag = false;
			/* cout << "MATR Q: " << endl;
			 double* checking_matr = new double[(size*amount_ev)];
			 Debuger_for_matr(matrix_w,size ,amount_ev, checking_matr);
			 cout << endl;*/
			 /*double * ch = new double[size];
		 Debuger(temp3, size, ch);
		 for(int i=0;i<size;i++)
		 cout << ch[i] << " " << endl;*/
		 }
		 

		 //All zero
		// *converge_temp = 0;
		 *b_i_old=0;
		 exit = false;
		 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (size*amount_ev, matrix_w);   //clear lanc vectors matr
		 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (amount_ev*amount_ev, matr_dense);  // clear matr T or eigen vectors matr
		 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (size*amount_ev, multi_temp);  // clear Mx matr
		 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (amount_ev-1, beta_vec);
		 all_zero << <10000, deviceProp.maxThreadsPerBlock >> > (amount_ev, alpha_vec);
		 double * ch = new double[amount_ev];
		 Debuger(converge_eig_val, amount_ev, ch);
		 for(int i=0;i<*tmp3;i++)
		 cout << "E_Values on iteration:  " << 1 / ch[i] << endl;
		   checkCudaErrors(cudaMemset(converge_temp, 0.0, sizeof(double))); 
		   
		   *CONVERGE_AMOUNT_CPU += *tmp3;
		   checkCudaErrors(cudaMemcpy(CONVERGE_AMOUNT, CONVERGE_AMOUNT_CPU, sizeof(int), cudaMemcpyHostToDevice));
		  // checkCudaErrors(cudaMemcpy(CONVERGE_AMOUNT_CPU, CONVERGE_AMOUNT, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
		   step++;
		   if(*tmp3==0)
		   {
			   cout << endl << "NO NEW CONVERGED EIGEN VALUES" << endl;
		   }
		   else
		   { first_flag = true;}
		   
		   if (step == 10)
			   break;
		  // *CONVERGE_AMOUNT_CPU = amount_ev;

	 }
	 cout << "STEP: "<<step << endl;
	 reverse_for_eigen_values_lanc << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_val, amount_ev);
	 checkCudaErrors(cudaMemcpy(eigen_values, converge_eig_val, sizeof(double)*(amount_ev), cudaMemcpyDeviceToHost));
	 //CUBLAS_OP_N - no				‘N’ or ‘n’
	 //CUBLAS_OP_T - yes			‘T’ or ‘t’
	 cout << "+++++++++FINAL SOLUTION ++++++++" << endl << endl << endl;
	 double* checking_matr = new double[amount_ev*size];
	 Debuger_for_matr(converge_eig_vec, size, amount_ev, checking_matr);
	 double * one_f = new double;
	 cout << "PROVERB ON RESULT:" << endl;
	 for (int j = 0; j < amount_ev; j++)
	 {
		 *one_f = -eigen_values[j];
		 return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (converge_eig_vec, j, size, temp3);
		 GPU_mult(temp3, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1_b, deviceProp.maxThreadsPerBlock);
		 GPU_mult(temp3, size, temp, diag_gpu, gpu, d_val, d_col, d_row, temp2, deviceProp.maxThreadsPerBlock);
		 checkCudaErrors(cublasDaxpy(cublasHandle, size, one_f, temp1_b, 1, temp2, 1));//end
		 checkCudaErrors(cublasDnrm2(cublasHandle,size,temp2,1,tmp1));
		 cout << "Nevazka:[" << j << "]= " << *tmp1<<endl;
	 }
	// *one_f = 0.098569;//1.786052;
	// cout << endl << "Mathcad check:" << endl;
	// cout << "For eigenvalue: " << *one_f << endl;
	// *one_f = -*one_f;
	// double * test = new double[size];
	// double * test_gpu;
	// test[0] = -1;//-0.127;
	// test[1] = -0.898;//0,006265;
	// test[2] = -0.372;//-1;
	// test[3] =0;//0.074;
	// test[4] =0;//0;
	// test[5] =0;//0;
	// test[6] =0;//0;
	// test[7] =0;//0;
	// test[8] =0;//0;
	// test[9] = 0.368;//0.569;
	// checkCudaErrors(cudaMalloc((void **)&test_gpu, sizeof(double)*(size)));
	// checkCudaErrors(cudaMemcpy(test_gpu, test, sizeof(double)*(size), cudaMemcpyHostToDevice));
	//// return_vec << <10000, deviceProp.maxThreadsPerBlock >> > (test_gpu, j, size, temp3);
	// GPU_mult(test_gpu, size2, temp_M, diag_gpu2, gpu, d_val2, d_col2, d_row2, temp1_b, deviceProp.maxThreadsPerBlock);
	// GPU_mult(test_gpu, size, temp, diag_gpu, gpu, d_val, d_col, d_row, temp2, deviceProp.maxThreadsPerBlock);
	// checkCudaErrors(cublasDaxpy(cublasHandle, size, one_f, temp1_b, 1, temp2, 1));//end
	// checkCudaErrors(cublasDnrm2(cublasHandle, size, temp2, 1, tmp1));
	// cout << "Nevazka:= " << *tmp1 << endl;





}


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}


