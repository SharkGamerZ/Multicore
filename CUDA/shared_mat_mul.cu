#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width);
__global__ void sharedMatrixMulKernel(float* M, float* N, float* P, int Width);

void initMat(float* M, int n, int init);
void printMat(float* M, int n);
void writeMat(float* M, int n);


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

	dim3 BlockSize(16,16,1);
	dim3 GridSize((n/16),(n/16),1);


	float *h_M = (float*)malloc((n*n)*sizeof(float));
	float *h_N = (float*)malloc((n*n)*sizeof(float));
	float *h_P = (float*)malloc((n*n)*sizeof(float));;

	initMat(h_M, n, init);
	initMat(h_N, n, init);


	float *d_M;
	float *d_N;
	float *d_P;
	cudaMalloc(&d_M, (n*n)*sizeof(float));
	cudaMemcpy(d_M, h_M, (n*n)*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&d_N, (n*n)*sizeof(float));
	cudaMemcpy(d_N, h_N, (n*n)*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&d_P, (n*n)*sizeof(float));


	MatrixMulKernel<<<GridSize,BlockSize>>>(d_M, d_N, d_P, n);


	cudaMemcpy(h_P, d_P, (n*n)*sizeof(float), cudaMemcpyDeviceToHost);

	writeMat(h_P, n);
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


void writeMat(float* M, int n) {
    FILE* file = fopen("result_matrix.txt", "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }


    fwrite(M, sizeof(float), n*n, file);

    fclose(file);
}