#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_WIDTH 16
#define SERIAL_OUT "mat_serial"
#define NORMAL_OUT "mat_normal"
#define SHARED_OUT "mat_shared"


void serialMatrixMul(float* M, float* N, float* P, int Width) ;
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width);
__global__ void sharedMatrixMulKernel(float* M, float* N, float* P, int Width);

void initMat(float* M, int n, int init);
void printMat(float* M, int n);
void writeMat(char* output, float* M, int n);


int main(int argc, char** argv) {
	int n, init;
	//Check for the input file and output file names
	switch(argc) {
		case 1:
			n = 16;
			init = 1;
			break;
		case 3:
			n = atoi(argv[1]);
			init = atoi(argv[2]);
            break;
		default:
			printf("Usage: <executable> matrix_size matrix_number\n");
			exit(1);
	}


	// Start of Serial part
	//
	//
	//

	// Allocating matrixes
	float *M = (float*)malloc((n*n)*sizeof(float));
	float *N = (float*)malloc((n*n)*sizeof(float));
	float *P = (float*)malloc((n*n)*sizeof(float));;

	// Initializing matrixes
	initMat(M, n, init);
	initMat(N, n, init);

	clock_t start, end;
	double cpu_time_used;	

	start = clock();
	serialMatrixMul(M, N, P, n);
	end = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("La versione seriale ha impiegato %lf\n", cpu_time_used);

	//writeMat(SERIAL_OUT,P, n);




	// Start of CUDA code
	//
	//
	//

	// Setting up variables for measuring time
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);
	float milliseconds;

	// Number of threads and blocks
	dim3 BlockSize(16,16,1);
	dim3 GridSize((n+15)/16,(n+15)/16,1);

	// Allocating host matrixes
	float *h_M = (float*)malloc((n*n)*sizeof(float));
	float *h_N = (float*)malloc((n*n)*sizeof(float));
	float *h_P = (float*)malloc((n*n)*sizeof(float));;

	// Initializing matrixes
	initMat(h_M, n, init);
	initMat(h_N, n, init);


	// Allocating and copying matrixes to device
	float *d_M;
	float *d_N;
	float *d_P;
	cudaMalloc(&d_M, (n*n)*sizeof(float));
	cudaMemcpy(d_M, h_M, (n*n)*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&d_N, (n*n)*sizeof(float));
	cudaMemcpy(d_N, h_N, (n*n)*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&d_P, (n*n)*sizeof(float));




	// Cuda Normal 
	//
	//
	cudaEventRecord(cudaStart);
	MatrixMulKernel<<<GridSize,BlockSize>>>(d_M, d_N, d_P, n);
	cudaEventRecord(cudaStop);

	cudaMemcpy(h_P, d_P, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);
    
	// Timing
    cudaEventSynchronize(cudaStop);
    milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
	printf("La versione normale ha impiegato %f \n", milliseconds);

	//writeMat(NORMAL_OUT,h_P, n);





	// Cuda Shared 
	//
	//
	cudaEventRecord(cudaStart);
	sharedMatrixMulKernel<<<GridSize,BlockSize>>>(d_M, d_N, d_P, n);
	cudaEventRecord(cudaStop);

	cudaMemcpy(h_P, d_P, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);

	// Timing
	cudaEventSynchronize(cudaStop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
	printf("La versione shared ha impiegato %f \n", milliseconds);
	

	//writeMat(SHARED_OUT,h_P, n);
}



/* Funzioni */

void serialMatrixMul(float* M, float* N, float* P, int Width) 
{
	for(int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {

			P[j + i*Width] = 0;
			for(int k = 0; k < Width; k++) {
				P[j + i * Width] += M[k + i*Width]*N[j + k*Width];
			}
		}
	}
}

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) 
{
	int c = threadIdx.x+blockIdx.x*blockDim.x;
	int r = threadIdx.y+blockIdx.y*blockDim.y;
	
	if(c >= Width || r >= Width) return;
	
	P[c + r*Width] = 0;

	for(int i = 0; i < Width; i++) {
		P[c + r * Width] += M[i + r*Width]*N[c + i*Width];
	}
}

__global__ void sharedMatrixMulKernel(float* M, float* N, float* P, int Width)
{
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	float Pvalue = 0;

	// Loop over the M and N tiles required to compute the P element
	for (int p = 0; p < Width/TILE_WIDTH; ++p) {
		// Collaborative loading of M and N tiles into shared memory
		ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
		ds_N[ty][tx] = N[(p*TILE_WIDTH+ty)*Width + Col];
		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)  
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		__syncthreads();
	}	
	P[Row*Width+Col] = Pvalue;
}

void initMat(float* M, int n, int init) {
	for(int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++){
			M[i*n+j] = init;
		}
	}
}

void printMat(float* M, int n) {
	for(int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++){
			printf("%g ",M[i*n+j]);
		}
		printf("\n");
	}
}


void writeMat(char* output, float* M, int n) {
    FILE* file = fopen(output, "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    for (int i = 0; i < n; i++) {
    	for (int j = 0; j < n; j++) {
    		fprintf(file, "%g ",M[i*n + j]);
    	}
    	fprintf(file,"\n");
    }


    fclose(file);
}