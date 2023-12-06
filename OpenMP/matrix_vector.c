#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>



void generate(int*a, int n);
void generate_s(int*a, int n);
void generateC(int*a, int n);
void generateC_s(int*a, int n);

int main(int argc, char const *argv[])
{
	double dtime;
	
	int n = 20000;
	int* a = malloc((n*n)*sizeof(int));

	dtime = omp_get_wtime();
	generate(a, n);
	dtime = omp_get_wtime() - dtime;

	printf("Ci ho messo %fs!\n",dtime);


	dtime = omp_get_wtime();
	generateC(a, n);
	dtime = omp_get_wtime() - dtime;

	printf("Ci ho messo %fs!\n",dtime);


	dtime = omp_get_wtime();
	generate_s(a, n);
	dtime = omp_get_wtime() - dtime;

	printf("Ci ho messo %fs!\n",dtime);

	dtime = omp_get_wtime();
	generateC_s(a, n);
	dtime = omp_get_wtime() - dtime;

	printf("Ci ho messo %fs!\n",dtime);





	return 0;
}

void generateC_s(int* a, int n) {

	#pragma omp parallel num_threads(4)
	{
		#pragma omp for collapse(2) schedule(static,8)
		for (int i = 0; i < n; i++) {	
			for (int j = 0; j < n; j++) {
				a[i*n+j] = 0;
			}
		}	
	}
}

void generate_s(int* a, int n) {

	#pragma omp parallel num_threads(4)
	{
		#pragma omp for schedule(static,8)
		for (int i = 0; i < n; i++) {	
			for (int j = 0; j < n; j++) {
				a[i*n+j] = 0;
			}
		}	
	}
}

void generateC(int* a, int n) {

	#pragma omp parallel num_threads(4)
	{
		#pragma omp for collapse(2)
		for (int i = 0; i < n; i++) {	
			for (int j = 0; j < n; j++) {
				a[i*n+j] = 0;
			}
		}	
	}
}

void generate(int* a, int n) {

	{
		for (int i = 0; i < n; i++) {	
			for (int j = 0; j < n; j++) {
				a[i*n+j] = 0;
			}
		}	
	}
}