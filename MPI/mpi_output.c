#include <stdio.h>
#include <mpi.h>

int main(int argc, char const *argv[])
{
	int vai = 1;

	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


	//Ogni processo, tranne l'ultimo, stampano un messaggio e poi inviano un dato al
	//processo successivo, per farlo andare
	if (my_rank == 0) {  
		printf("[Processo %d] Ciao!\n", my_rank);
		MPI_Send(&vai, 1, MPI_INT, my_rank+1, 0, MPI_COMM_WORLD);
	} else if (my_rank != comm_sz -1) {
		MPI_Recv(&vai, 1, MPI_INT, my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[Processo %d] Ciao!\n", my_rank);
		MPI_Send(&vai, 1, MPI_INT, my_rank+1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(&vai, 1, MPI_INT, my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[Processo %d] Ciao!\n", my_rank);
	}



	MPI_Finalize();
	return 0;
}