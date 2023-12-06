#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

double genera(int a,int b) {
	double range = (b-a);
	double div = RAND_MAX / range;
	return a + (rand() / div);
}

int main(int argc, char const *argv[])
{
	int my_rank, comm_sz;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	

	srand(time(NULL));
	int number_of_tosses = 1000000, number_in_circle;
	double my_x, my_y, my_distance_squared;

	

	int my_number_of_tosses = number_of_tosses/comm_sz;


	int my_number_in_circle = 0;
	for (int my_toss = 0; my_toss < my_number_of_tosses; my_toss++) {
		my_x = genera(-1,1);
		my_y = genera(-1,1);

		my_distance_squared = my_x*my_x + my_y*my_y;
		if (my_distance_squared <= 1) my_number_in_circle++;
		//printf("[Processo %d] x:%f y:%f distance_squared:%f\n",my_rank, my_x, my_y, my_distance_squared);
		
	}
	

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

	
	MPI_Finalize();
	return 0;
}