#include <stdio.h>
#include <sys/time.h>

//number of channels i.e. R G B
#define CHANNELS 3

unsigned char* loadPPM(const char* path, int* width, int* height);
void writePPM(const char* path, unsigned char* img, int width, int height);

void serial_blur(unsigned char* rgb_image,unsigned char*blur_image, int rows, int cols, int bsize)  {
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

__global__ 	void kernel_blur(unsigned char* d_rgb_image,unsigned char*d_blur_image, int rows,int cols,int bsize) {
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

int main(int argc, char **argv) 
{	
	char* input_file;
	char* output_file;
	char* tmp;
    int bsize=4;
	//Check for the input file and output file names
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



	
	int rows; //number of rows of pixels
	int cols; //number of columns of pixels
	



	// Start of Serial part
	//
	//
	//
	clock_t start, end;
	double cpu_time_used;		


	unsigned char *s_rgb_image; //store image's rbg data
	unsigned char *s_blur_image; //array for storing rgb data on device

	//load image into an array and retrieve number of pixels
	s_rgb_image = loadPPM(input_file, &cols, &rows); 
	if (s_rgb_image == NULL) return -1;
	
	int total_pixels=rows*cols;

	//allocate memory of host's blur image data array
	s_blur_image = (unsigned char *)malloc(sizeof(unsigned char*) * total_pixels * CHANNELS);


	start = clock();
	serial_blur(s_rgb_image, s_blur_image, rows, cols, bsize);
	end = clock();

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Seriale ha impiegato %lf secondi\n", cpu_time_used);

	//output the blurred image
	tmp = (char*) malloc((strlen(output_file)+10) * sizeof(char));
	strcpy(tmp, output_file);
    writePPM(strcat(tmp,"_serial.ppm"), s_blur_image, cols, rows);
    free(tmp);

	




	// Start of CUDA part
	//
	//
	//
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

    dim3 BlockSize(16,16,1);
	dim3 GridSize((cols/16)+1,(rows/16)+1,1);


	unsigned char *h_rgb_image; //store image's rbg data
	unsigned char *d_rgb_image; //array for storing rgb data on device
	unsigned char *h_blur_image, *d_blur_image; //host and device's blur image data array pointers

	h_rgb_image = loadPPM(input_file, &cols, &rows);

	if (h_rgb_image == NULL) return -1;

	//allocate memory of host's blur image data array
	h_blur_image = (unsigned char *)malloc(sizeof(unsigned char*) * total_pixels * CHANNELS);

	cudaMalloc(&d_rgb_image,total_pixels*CHANNELS*8);
	cudaMemcpy(d_rgb_image,h_rgb_image,total_pixels*CHANNELS*8,cudaMemcpyHostToDevice);

	cudaMalloc(&d_blur_image,total_pixels*CHANNELS*8);


	cudaEventRecord(cudaStart);
	kernel_blur<<<GridSize,BlockSize>>>(d_rgb_image,d_blur_image,rows,cols,bsize);
	
	

    cudaMemcpy(h_blur_image,d_blur_image,total_pixels*CHANNELS*8,cudaMemcpyDeviceToHost);
	
    cudaDeviceSynchronize();
    cudaEventRecord(cudaStop);
    
	cudaEventSynchronize(cudaStop);
    	

    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);

	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("CUDA ha impiegato %f o %lf secondi\n", milliseconds, cpu_time_used);

	// Free space
	cudaFree(d_rgb_image);
	cudaFree(d_blur_image);
	
    //output the blurred image
	tmp = (char*) malloc((strlen(output_file) + 10) * sizeof(char));
	strcpy(tmp, output_file);
    writePPM(strcat(tmp,"_cuda.ppm"), h_blur_image, cols, rows);
    free(tmp);




	return 0;
}

//function for loading an image into rgb format unsigned char array

unsigned char* loadPPM(const char* path, int* width, int* height) {
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

//function for writing blurred image data array to the image file
void writePPM(const char* path, unsigned char* img, int width, int height) {
    FILE* file = fopen(path, "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    fwrite(img, CHANNELS, width * height, file);

    fclose(file);
}
