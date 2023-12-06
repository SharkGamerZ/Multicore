#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>



int main(int argc, char const *argv[])
{
	int my_rank, comm_sz;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	srand(time(NULL));


	int n = 16;
	int v[n];
	if(my_rank == 0) {
		//Genera vettore 
		for (int i = 0; i < n; i++) {
			v[i] = (rand()%2000) - 1000;	// Genera numeri tra -1000 e +1000
		}	
	}

	int my_n = n/comm_sz;

	int my_start = my_rank*my_n;
	int my_end = my_start + my_n - 1;

	if (n%comm_sz != 0) {
		if (my_rank == comm_sz - 1) {
			my_end = n-1;
			my_n += n%comm_sz;
		}
	}

	for (int i = my_start; i < my_end; i+=2) {
		
	}




	printf("[Processo %d] my_start = %d, my_end =%d, my_n = %d\n",my_rank, my_start, my_end, my_n );


	/*
	if (my_rank != 0) {
		MPI_Send(&my_number_in_circle, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	} else {
		number_in_circle = my_number_in_circle;
		for (int i = 1; i < comm_sz; i++) {
			MPI_Recv(&my_number_in_circle, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			number_in_circle += my_number_in_circle;
		}

		double pi_estimate = 4*number_in_circle/ (double) number_of_tosses;
	
		printf("%f\n",pi_estimate );
	}

	*/
	MPI_Finalize();
	return 0;
}