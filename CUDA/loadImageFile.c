#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHANNELS 3


unsigned char* loadImage(const char* input_file, size_t *num_of_pixels, int *rows, int *cols);
void outputImage(const char* output_file, unsigned char* blur_image, int rows, int cols);

int main(int argc, char **argv)
{
    char* input_file;
    char* output_file;
    int bsize;

    //Check for the input file and output file names
    switch(argc) {
        case 3:
            input_file = argv[1];
            output_file = argv[2];
            bsize = 4;
            break;
        case 4:
            input_file = argv[1];
            output_file = argv[2];
            bsize = atoi(argv[3]);
            break;
        default:
            fprintf(stderr,"Usage: <executable> input_file output_file blur_size");
            exit(1);
    }

    unsigned char *h_rgb_image; //store image's rbg data
    unsigned char *d_rgb_image; //array for storing rgb data on device
    unsigned char *h_blur_image, *d_blur_image; //host and device's blur image data array pointers
    int rows; //number of rows of pixels
    int cols; //number of columns of pixels

    //load image into an array and retrieve number of pixels
    size_t total_pixels;
    h_rgb_image = loadImage(input_file, &total_pixels, &rows, &cols);

    //allocate memory of host's blur image data array
    h_blur_image = (unsigned char *)malloc(sizeof(unsigned char*) * total_pixels * CHANNELS);


    outputImage(output_file, h_rgb_image, rows, cols);
}



unsigned char* loadImage(const char* input_file, size_t *num_of_pixels, int *rows, int *cols)
{
    char buff[16];
    FILE *fp;

    unsigned char* img;
    int c, rgb_comp_color;


    // Apro il file PPM
    fp = fopen(input_file, "rb");
    if (!fp) {  //Se il file non può essere aperto
        fprintf(stderr, "Unable to open file '%s'\n", input_file);
        exit(1);
    }


    // Legge il formato
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(input_file);
        exit(1);
    }
    // Vede se il formato è "P6"
    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }


    // Controlla se ci sono commenti
    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n') ;
        c = getc(fp);
    }
    ungetc(c, fp);


    // Legge la larghezza e l'altezza
    if (fscanf(fp, "%d %d", cols, rows) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", input_file);
        exit(1);
    }


    // Legge la componente RGB
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", input_file);
        exit(1);
    }
    // Controlla che sia uno spazio colori ad 8 bit
    if (rgb_comp_color != 255) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", input_file);
        exit(1);
    }

    // Arriviamo al line feed
    while (fgetc(fp) != '\n');


    // Alloca la memoria per l'immagine
    img = malloc(*rows * *cols * CHANNELS);
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    //read pixel data from file
    if (fread(img, 3 * *cols, *rows, fp) != *rows) {
        fprintf(stderr, "Error loading image '%s'\n", input_file);
        exit(1);
    }

    fclose(fp);
    return img;
}


void outputImage(const char* output_file, unsigned char* blur_image, int rows, int cols)
{
    FILE *fp;

    // Apro il file PPM
    fp = fopen(output_file, "w");
    if (!fp) {  //Se il file non può essere aperto
        fprintf(stderr, "Unable to open file '%s'\n", output_file);
        exit(1);
    }


    // Scrive il formato
    fprintf(fp,"P6\n");

    // Scrive la larghezza e l'altezza
    fprintf(fp, "%d %d\n", cols, rows);


    // Scrive la componente RGB
    fprintf(fp, "%d\n", 255);


    // Scrive i pixel
    for (int i = 0; i < 3*rows*cols; i++) {
        fprintf(fp, "%c", blur_image[i]);
    }
    
    fclose(fp);
}