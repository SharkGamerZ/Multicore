/* DA IMPLEMENTARE
	[ ] Far copiare i dati sulla shared memori da ogni thread
	[ ] Far lavorare solo i thread interni per fare il blur
	[ ] Cambio della tileSize in base alla grandezz del blur




*/
#include <stdio.h>
#include <sys/time.h>
#include "../ppm.h"

//number of channels i.e. R G B
#define CHANNELS 3
#define TILE_WIDTH 16

// Macro to check errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




void serialBlur					(unsigned char* rgb_image,unsigned char*blur_image, int rows, int cols, int bsize);
__global__ void kernelBlur 		(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows, int cols, int bsize);
__global__ void sharedKernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows, int cols, int bsize, int tileSize);


int main(int argc, char **argv) 
{	
	char* input_file;
	char* output_file;
	char* tmp;
    int bsize=4;

	// Check for the input file and output file names
	switch(argc) {
		case 4:
			input_file = argv[1];
			output_file = argv[2];
			bsize = atoi(argv[3]);
			if (bsize < 0) {
				fprintf(stderr, "bsize can't be < 0\n");
				exit(1);
			}
            break;
		case 3:
			input_file = argv[1];
			output_file = argv[2];
            break;
		default:
			printf("Usage: <executable> input_file output_file bsize\n");
			exit(1);
	}

	int tileHalo = TILE_WIDTH + bsize*2;

	// Image dimensions
	int rows;
	int cols;
	loadPPM(input_file, &cols, &rows); 


    dim3 BlockSize(16, 16, 1);
    dim3 BlockSizeShared(tileHalo, tileHalo, 1);
	dim3 GridSize((cols+15)/16, (rows+15)/16, 1);


	// Serial part
	//
	//
	//

	if (false)
	{	
		// Timing variables
		clock_t start, end;
		double cpu_time_used;


		// Declaring serial RGB and blur images
		unsigned char *s_rgb_image;
		unsigned char *s_blur_image;
	
		// Load image from file and calculate size
		s_rgb_image = loadPPM(input_file, &cols, &rows); 
		if (s_rgb_image == NULL) return -1;
		int total_pixels=rows*cols;
	
		// Allocate memory for blur image
		s_blur_image = (unsigned char *)malloc(total_pixels * CHANNELS);
	
	
		start = clock();
		serialBlur(s_rgb_image, s_blur_image, rows, cols, bsize);
		end = clock();
	
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Seriale ha impiegato %lf secondi\n", cpu_time_used);
	
		// Write the blurred image to file
		tmp = (char*) malloc((strlen(output_file)+10) * sizeof(char));
		strcpy(tmp, output_file);
	    writePPM(strcat(tmp,"_serial.ppm"), s_blur_image, cols, rows);
	    free(tmp);
	
	    free(s_rgb_image);
	    free(s_blur_image);
	}


	// CUDA Normal
	//
	//
	//
	if (false)
	{
		// Timing variables
		cudaEvent_t cudaStart, cudaStop;	
		float milliseconds;

		cudaEventCreate(&cudaStart);
		cudaEventCreate(&cudaStop);




		// Declaring host RGB and blur images
		unsigned char *h_rgb_image;
		unsigned char *h_blur_image;

		// Declaring device RGB and blur images
		unsigned char *d_rgb_image;
		unsigned char *d_blur_image;


		// Loading host RGB image
		h_rgb_image = loadPPM(input_file, &cols, &rows);
		if (h_rgb_image == NULL) return -1;
		int total_pixels=rows*cols;

		// Allocate host blur image memory
		h_blur_image = (unsigned char *)malloc(total_pixels * CHANNELS);

		

		// Allocate device RGB and blur images and copying RGB image
		cudaMalloc(&d_rgb_image,total_pixels*CHANNELS);
		cudaMemcpy(d_rgb_image,h_rgb_image,total_pixels*CHANNELS,cudaMemcpyHostToDevice);
		cudaMalloc(&d_blur_image,total_pixels*CHANNELS);


		// Executing code
		cudaEventRecord(cudaStart);
		kernelBlur<<<GridSize,BlockSize>>>(d_rgb_image,d_blur_image,rows,cols,bsize);
	    cudaEventRecord(cudaStop);

	    cudaMemcpy(h_blur_image,d_blur_image,total_pixels*CHANNELS,cudaMemcpyDeviceToHost);

	    // Timing
		cudaEventSynchronize(cudaStop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
		printf("CUDA normale ha impiegato %f\n", milliseconds);
		
	    // Output the blurred image
		tmp = (char*) malloc((strlen(output_file) + 10) * sizeof(char));
		strcpy(tmp, output_file);
	    writePPM(strcat(tmp,"_cuda.ppm"), h_blur_image, cols, rows);
	    free(tmp);

	    cudaFree(d_rgb_image);
	    cudaFree(d_blur_image);
	}

	// CUDA Shared
	// 
	// 
	// 
	// 
	if (true)
	{
		// Timing variables
		cudaEvent_t cudaStart, cudaStop;	
		float milliseconds;
		cudaEventCreate(&cudaStart);
		cudaEventCreate(&cudaStop);


		// Declaring host RGB and blur images
		unsigned char *h_rgb_image;
		unsigned char *h_blur_image;

		// Declaring device RGB and blur images
		unsigned char *d_rgb_image;
		unsigned char *d_blur_image;


		// Loading host RGB image
		h_rgb_image = loadPPM(input_file, &cols, &rows);
		if (h_rgb_image == NULL) return -1;
		int total_pixels=rows*cols;

		// Allocate host blur image memory
		h_blur_image = (unsigned char *)malloc(total_pixels * CHANNELS);

		

		// Allocate device RGB and blur images and copying RGB image
		cudaMalloc(&d_rgb_image,total_pixels*CHANNELS);
		cudaMemcpy(d_rgb_image,h_rgb_image,total_pixels*CHANNELS,cudaMemcpyHostToDevice);
		cudaMalloc(&d_blur_image,total_pixels*CHANNELS);

		// Executing code
		cudaEventRecord(cudaStart);
		sharedKernelBlur<<<GridSize, BlockSizeShared, ((tileHalo*tileHalo) * CHANNELS)>>>(d_rgb_image, d_blur_image, rows, cols, bsize, TILE_WIDTH);
	    cudaEventRecord(cudaStop);

	    cudaMemcpy(h_blur_image, d_blur_image, total_pixels*CHANNELS, cudaMemcpyDeviceToHost);

	    // Timing
		cudaEventSynchronize(cudaStop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
		printf("CUDA shared ha impiegato %f\n", milliseconds);
		
	    // Output the blurred image
		tmp = (char*) malloc((strlen(output_file) + 10) * sizeof(char));
		strcpy(tmp, output_file);
	    writePPM(strcat(tmp,"_shared.ppm"), h_blur_image, cols, rows);
	    free(tmp);

	    cudaFree(d_rgb_image);
	    cudaFree(d_blur_image);
	}


	printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	return 0;
}


void serialBlur(unsigned char* rgb_image,unsigned char*blur_image, int rows, int cols, int bsize)
{
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			unsigned int red  =0;
		    unsigned int green=0;
		    unsigned int blue =0;
		    int num=0; 
		    int curr_i, curr_j;
			for (int m = -bsize; m <= bsize; m++) {
				for (int n = -bsize; n <= bsize; n++) {
					curr_i = i + m;
					curr_j = j + n;
					if((curr_i<0)||(curr_i>rows-1)||(curr_j<0)||(curr_j>cols-1)) continue; 
					red   += rgb_image[(3*(curr_j+curr_i*cols))];
					green += rgb_image[(3*(curr_j+curr_i*cols))+1];
					blue  += rgb_image[(3*(curr_j+curr_i*cols))+2];
					num++;
				}
			}

			red /= num;
			green /= num;
			blue /= num;


			blur_image[3*(j+i*cols)]	=red;
		    blur_image[3*(j+i*cols)+1]=green;
		    blur_image[3*(j+i*cols)+2]=blue;

		}
	}
}

__global__ 	void kernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int bsize)
{
	int c = threadIdx.x+blockIdx.x*blockDim.x;
	int r = threadIdx.y+blockIdx.y*blockDim.y;

	if(c >= cols || r >= rows) return;

	unsigned int red  =0;
	unsigned int green=0;
	unsigned int blue =0;
	int num=0; 

	int curr_c;
	int curr_r;

	for (int i = -bsize; i <= bsize; i++)
		for (int j = -bsize; j <= bsize; j++) {
			curr_c = c + i;
			curr_r = r + j;
			if((curr_r<0)||(curr_r>rows-1)||(curr_c<0)||(curr_c>cols-1)) continue; 
			red   += d_rgb_image[(3*(curr_c+curr_r*cols))];
			green += d_rgb_image[(3*(curr_c+curr_r*cols))+1];
			blue  += d_rgb_image[(3*(curr_c+curr_r*cols))+2];
			num++;
			}
	red /= num;
	green /= num;
	blue /= num;


	d_blur_image[3*(c+r*cols)]		= red;
	d_blur_image[3*(c+r*cols)+1]	= green;
	d_blur_image[3*(c+r*cols)+2]	= blue;
}


__global__ 	void sharedKernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows, int cols, int bsize, int tileSize)
{
	extern __shared__ unsigned char ds_rgb_image[];

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int g_c = bx*(blockDim.x - 2*bsize) + tx - bsize;
	int g_r = by*(blockDim.y - 2*bsize) + ty - bsize;


	int c = tx - bsize;
	int r = ty - bsize;

	// If the thread it's out of the image
	if(g_c < 0 || g_c >= cols || g_r < 0 || g_r >= rows) return;

	
	unsigned int red  	=	0;
	unsigned int green	=	0;
	unsigned int blue 	=	0;
   int num=0; 

	ds_rgb_image[3*(ty*blockDim.x + tx)] 		= d_rgb_image[(3*(g_c + g_r*cols))];
	ds_rgb_image[3*(ty*blockDim.x + tx) + 1] 	= d_rgb_image[(3*(g_c + g_r*cols)) + 1];
	ds_rgb_image[3*(ty*blockDim.x + tx) + 2] 	= d_rgb_image[(3*(g_c + g_r*cols)) + 2];

	__syncthreads();
	// If the thread it's in the halo
	if (c < 0 || c > (tileSize + 2*bsize) || r < 0 || r > (tileSize + 2*bsize)) return;

	int curr_c;
	int curr_r;

	if((bx != 0 || by != 0) && (tx != 0 || ty != 0)) return;
    
	for (int i = -bsize; i <= bsize; i++) {
		for (int j = -bsize; j <= bsize; j++) {
			curr_c = g_c + i;
			curr_r = g_r + j;

			if((curr_r + j<0)||(curr_r + j>rows-1)||(curr_c + i<0)||(curr_c + i>cols-1)) continue;
			red   += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i))    ];
			green += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i)) + 1];
			blue  += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i)) + 2];

			printf("i=%d j=%d red=%d\n", i, j, red);

			num++;
		}
	}
	
	printf("uscito\n");
   

	red /= num;
	green /= num;
	blue /= num;
	
	printf("tx=%d ty=%d   mem=%d\n", tx, ty, 3*(g_c + g_r*cols));

	printf("%d\n", red);
	return;

	d_blur_image[0] = red;

	return;

	d_blur_image[3*(g_c + g_r*cols)]		= red; 	//ds_rgb_image[3*(ty*blockDim.x + tx)];
	d_blur_image[3*(g_c + g_r*cols)+1]	= green; //ds_rgb_image[3*(ty*blockDim.x + tx)+1];
	d_blur_image[3*(g_c + g_r*cols)+2]	= blue; 	//ds_rgb_image[3*(ty*blockDim.x + tx)+2];
}
