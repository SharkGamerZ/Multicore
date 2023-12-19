#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

//number of channels i.e. R G B
#define CHANNELS 3
#define TILE_WIDTH 16


__global__ 	void kernelGrayScale(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int grayLevel);


unsigned char* loadPPM(const char* path, int* width, int* height);
void writePPM(const char* path, unsigned char* img, int width, int height);

int main(int argc, char **argv) 
{	
	char* input_file;
	char* output_file;
    int grayLevel = 0;

	// Check for the input file and output file names
	switch(argc) {
		case 4:
			input_file = argv[1];
			output_file = argv[2];
			grayLevel = atoi(argv[3]);
            break;
		case 3:
			input_file = argv[1];
			output_file = argv[2];
            break;
		default:
			fprintf(stderr, "Usage: <executable> input_file output_file grayLevel\n");
			exit(1);
	}

	if(grayLevel < 0 || grayLevel > 7 ) {
		fprintf(stderr, "grayLevel must be between 1 and 7\n");
		exit(1);
	}


	// Image dimensions
	int rows;
	int cols;
	
	// Timing variables
	cudaEvent_t cudaStart, cudaStop;	
	float milliseconds;

	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);


	// Declaring host RGB and gray images
	unsigned char *h_rgb_image;
	unsigned char *h_gray_image;

	// Declaring device RGB and gray images
	unsigned char *d_rgb_image;
	unsigned char *d_gray_image;


	// Loading host RGB image
	h_rgb_image = loadPPM(input_file, &cols, &rows);
	if (h_rgb_image == NULL) return -1;
	int total_pixels = cols * rows;

	// Allocate host blur image memory
	h_gray_image = (unsigned char *)malloc(total_pixels*CHANNELS);

	
	// Number of threads
    dim3 BlockSize(16,16,1);
	dim3 GridSize((cols/16)+1,(rows/16)+1,1);

	// Allocate device RGB and blur images and copying RGB image
	cudaMalloc(&d_rgb_image,total_pixels*CHANNELS);
	cudaMemcpy(d_rgb_image,h_rgb_image,total_pixels*CHANNELS,cudaMemcpyHostToDevice);
	cudaMalloc(&d_gray_image,total_pixels*CHANNELS);


	// Executing
	cudaEventRecord(cudaStart);
	kernelGrayScale<<<GridSize,BlockSize>>>(d_rgb_image, d_gray_image, rows, cols, grayLevel);
    cudaEventRecord(cudaStop);

    cudaMemcpy(h_gray_image,d_gray_image,total_pixels*CHANNELS,cudaMemcpyDeviceToHost);

    // Timing
	cudaEventSynchronize(cudaStop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
	printf("Il programma ha impiegato %f\n", milliseconds);
	
    // Output the gray image
    writePPM(output_file, h_gray_image, cols, rows);


	// Free space
	cudaFree(d_rgb_image);
	cudaFree(d_gray_image);

	return 0;
}


__global__ 	void kernelGrayScale(unsigned char* d_rgb_image,unsigned char*d_gray_image, int rows, int cols, int grayLevel)
{
	int c = threadIdx.x+blockIdx.x*blockDim.x;
	int r = threadIdx.y+blockIdx.y*blockDim.y;
	
	if(c >= cols || r >= rows) return;

	float R_Weight = 0.2126;
	float G_Weight = 0.7152;
	float B_Weight = 0.0722;

    int value = 0; 


	value += d_rgb_image[(3*(c+r*cols))] * R_Weight;
	value += d_rgb_image[(3*(c+r*cols))+1]* G_Weight;
	value += d_rgb_image[(3*(c+r*cols))+2]* B_Weight;


	if (c == 200 && r == 0) printf("value = %u\n", value);

	value = value >> grayLevel;

	if (c == 200 && r == 0) printf("value >> %d = %u\n", grayLevel, value);

	// value = ( (value << grayLevel) | ((1 << grayLevel)-1));

	value = (value << grayLevel) *2 - 1;

	if (c == 200 && r == 0) printf("value << %d = %u\n", grayLevel, value);


	// value = (int) ((value * 1.0) / (((1 << 8) - 1) * 1.0) * (((1 << grayLevel) - 1)*1.0) + 0.5);

	d_gray_image[(3*(c+r*cols))] = value;
	d_gray_image[(3*(c+r*cols))+1] = value;
	d_gray_image[(3*(c+r*cols))+2] = value;
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
