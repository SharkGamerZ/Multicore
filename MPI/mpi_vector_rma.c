#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
	int my_rank, comm_sz;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	int MY_N = 10;

	int n = MY_N*comm_sz;

	int v1[n];
	int v2[n];

	int scalare;

	int* buf = NULL;
	MPI_Win win;


	if (my_rank == 0) {
		buf = malloc(2*n*sizeof(int));

		//printf("Primo vettore:\n");
		for (int i = 0; i < n; i++) {
			/*
			printf("Inserire il %d° numero: ",i+1);
			fflush( stdout );
			scanf("%lf\n",&v1[i]);
			*/
			buf[i] = rand()%10;
		}
		//printf("Secondo vettore:\n");
		for (int i = 0; i < n; i++) {
			/*
			printf("Inserire il %d° numero: ",i+1);
			fflush( stdout );
			scanf("%lf\n",&v2[i]);
			*/
			buf[n+i] = rand()%10;
		}

		printf("Inserire lo scalare: ");
		fflush(stdout);
		scanf("%lf",&scalare);

		
		MPI_Win_create(buf, 2*n*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

	}

	else { MPI_Win_create(NULL, 2*n*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win); }

	int my_v1[MY_N]; 
	int my_v2[MY_N]; 

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Get(my_v1, MY_N, MPI_INT, 0, my_rank*MY_N, MY_N, MPI_INT, win);
	MPI_Get(my_v2, MY_N, MPI_INT, 0, my_rank*MY_N+n, MY_N, MPI_INT, win);

	int prodotto_parziale = 0;

	// Moltiplico per scalare e calcolo c-dot product parziale
	for (int i = 0; i < MY_N; i++) {
		my_v1[i] *= scalare;
		my_v2[i] *= scalare;

		prodotto_parziale += my_v1[i]*my_v2[i];

		printf("[Processo %d]: V1:%d  V2:%d\n",my_rank, my_v1[i], my_v2[i]);
	}

	int prodotto_finale;

	MPI_Reduce(&prodotto_parziale,&prodotto_finale,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) printf("Prodotto = %d\n", prodotto_finale);




	MPI_Win_free(&win);

	MPI_Finalize();
	return 0;
}