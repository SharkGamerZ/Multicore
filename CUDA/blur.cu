#include <stdio.h>
#include <sys/time.h>

//number of channels i.e. R G B
#define CHANNELS 3
#define TILE_WIDTH 16


void serialBlur(unsigned char* rgb_image,unsigned char*blur_image, int rows, int cols, int bsize);
__global__ 	void kernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int bsize);
__global__ 	void sharedKernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int bsize);


unsigned char* loadPPM(const char* path, int* width, int* height);
void writePPM(const char* path, unsigned char* img, int width, int height);

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
            break;
		case 3:
			input_file = argv[1];
			output_file = argv[2];
			bsize = 4;
            break;
		default:
			printf("Usage: <executable> input_file output_file bsize\n");
			exit(1);
	}



	// Image dimensions
	int rows;
	int cols;
	
	// Timing variables
	clock_t start, end;
	double cpu_time_used;

	cudaEvent_t cudaStart, cudaStop;	
	float milliseconds;

	// Start of Serial part
	//
	//
	//

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





	// Start of CUDA part
	//
	//
	//
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);


// 
// 
// 
// 
// 
//    AGGIUSTARE NON FUNZIONA 
//      inserire bestemmia
// 
// 
// 
// 

    dim3 BlockSize(16,16,1);
	dim3 GridSize((cols/16)+1,(rows/16)+1,1);

	// Declaring host RGB and blur images
	unsigned char *h_rgb_image;
	unsigned char *h_blur_image;

	// Declaring device RGB and blur images
	unsigned char *d_rgb_image;
	unsigned char *d_blur_image;


	// Loading host RGB image
	h_rgb_image = loadPPM(input_file, &cols, &rows);
	if (h_rgb_image == NULL) return -1;

	// Allocate host blur image memory
	h_blur_image = (unsigned char *)malloc(total_pixels * CHANNELS);

	

	// Allocate device RGB and blur images and copying RGB image
	cudaMalloc(&d_rgb_image,total_pixels*CHANNELS);
	cudaMemcpy(d_rgb_image,h_rgb_image,total_pixels*CHANNELS,cudaMemcpyHostToDevice);
	cudaMalloc(&d_blur_image,total_pixels*CHANNELS);



	// Cuda Normal 
	//
	//
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




	// Cuda Shared 
	//
	//
	cudaEventRecord(cudaStart);
	sharedKernelBlur<<<GridSize,BlockSize>>>(d_rgb_image,d_blur_image,rows,cols,bsize);
    cudaEventRecord(cudaStop);

    cudaMemcpy(h_blur_image,d_blur_image,total_pixels*CHANNELS,cudaMemcpyDeviceToHost);

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






	// Free space
	cudaFree(d_rgb_image);
	cudaFree(d_blur_image);

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


	d_blur_image[3*(c+r*cols)]	=red;
    d_blur_image[3*(c+r*cols)+1]=green;
    d_blur_image[3*(c+r*cols)+2]=blue;
}


__global__ 	void sharedKernelBlur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int bsize)
{
	__shared__ unsigned char ds_rgb_image[(TILE_WIDTH * TILE_WIDTH) * CHANNELS];


	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int g_c = bx*blockDim.x + tx;
	int g_r = by*blockDim.y + ty;
	

	if(g_c >= cols || g_r >= rows) return;

    unsigned int red  =0;
    unsigned int green=0;
    unsigned int blue =0;
    int num=0; 

    ds_rgb_image[3*(ty*blockDim.x + tx)] 		= d_rgb_image[(3*(g_c + g_r*cols))];
    ds_rgb_image[3*(ty*blockDim.x + tx) + 1] 	= d_rgb_image[(3*(g_c + g_r*cols)) + 1];
    ds_rgb_image[3*(ty*blockDim.x + tx) + 2] 	= d_rgb_image[(3*(g_c + g_r*cols)) + 2];



    
	for (int i = -bsize; i <= bsize; i++)
		for (int j = -bsize; j <= bsize; j++) {
			//if((r + j<0)||(r + j>rows-1)||(g_c + i<0)||(g_c + i>cols-1)) continue; 
			if((ty+j) < 0 || (ty+j) >= blockDim.y || (tx+i) < 0 || (tx+i) >= blockDim.x) continue;
			red   += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i))];
			green += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i)) + 1];
			blue  += ds_rgb_image[3*((ty + j)*blockDim.x + (tx + i)) + 2];
			num++;
			}
	red /= num;
	green /= num;
	blue /= num;
	

	d_blur_image[3*(g_c + g_r*cols)]	= red; //ds_rgb_image[3*(ty*blockDim.x + tx)];
    d_blur_image[3*(g_c + g_r*cols)+1]	= green; //ds_rgb_image[3*(ty*blockDim.x + tx)+1];
    d_blur_image[3*(g_c + g_r*cols)+2]	= blue; //ds_rgb_image[3*(ty*blockDim.x + tx)+2];
}



// Function to load the image from file
unsigned char* loadPPM(const char* path, int* width, int* height)
{
	FILE* file = fopen(path, "rb");

	if (!file) {
		fprintf(stderr, "Failed to open file\n");
		return NULL;
	}

	char header[3];
	fscanf(file, "%2s", header);
	if (header[0] != 'P' || header[1] != '6') {
		fprintf(stderr, "Invalid PPM file\n");
		return NULL;
	}

	fscanf(file, "%d %d", width, height);

	int maxColor;
	fscanf(file, "%d", &maxColor);

	fgetc(file);  // Skip single whitespace character

	unsigned char* img = (unsigned char*) malloc((*width) * (*height) * CHANNELS);
	if (!img) {
		fprintf(stderr, "Failed to allocate memory\n");
		return NULL;
	}

	fread(img, CHANNELS, *width * *height, file);

	fclose(file);

	return img;
}

// Function to write the matrix image to file
void writePPM(const char* path, unsigned char* img, int width, int height)
{
    FILE* file = fopen(path, "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    fwrite(img, CHANNELS, width * height, file);

    fclose(file);
}
