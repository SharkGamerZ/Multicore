	#include <stdio.h>
#include <string.h>
#include <mpi.h>


const int MAX_STRING = 100;

int main() {

	char messaggio[MAX_STRING];
	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank != 0) { //Tutti i processi tranne 0
		sprintf(messaggio, "Sono il %d° di %d", my_rank, comm_sz);
		MPI_Send(messaggio, strlen(messaggio) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

	} else { //Il processo master
		printf("Sono il %d° di %d\n", my_rank, comm_sz);
		for (int i = 1; i < comm_sz; i++) {
			MPI_Recv(messaggio, MAX_STRING, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%s\n", messaggio);
		}
	}

	MPI_Finalize();
	return 0;

}