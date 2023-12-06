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

	int num_for_process = 10;

	int n = num_for_process*comm_sz;

	double v1[n];
	double v2[n];

	double scalare;

	if (my_rank == 0) {
		//printf("Primo vettore:\n");
		for (int i = 0; i < n; i++) {
			/*
			printf("Inserire il %d° numero: ",i+1);
			fflush( stdout );
			scanf("%lf\n",&v1[i]);
			*/
			v1[i] = rand()%10;
		}
		//printf("Secondo vettore:\n");
		for (int i = 0; i < n; i++) {
			/*
			printf("Inserire il %d° numero: ",i+1);
			fflush( stdout );
			scanf("%lf\n",&v2[i]);
			*/
			v2[i] = rand()%10;
		}

		printf("Inserire lo scalare: ");
		fflush(stdout);
		scanf("%lf",&scalare);

		
	}

	double my_v1[num_for_process]; 
	double my_v2[num_for_process]; 
	MPI_Scatter(v1,num_for_process, MPI_DOUBLE, my_v1, num_for_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(v2,num_for_process, MPI_DOUBLE, my_v2, num_for_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&scalare, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double prodotto_parziale = 0;

	// Moltiplico per scalare e calcolo c-dot product parziale
	for (int i = 0; i < num_for_process; i++) {
		my_v1[i] *= scalare;
		my_v2[i] *= scalare;

		prodotto_parziale += my_v1[i]*my_v2[i];

		printf("[Processo %d]: V1:%d  V2:%d\n",my_rank, (int)my_v1[i], (int) my_v2[i]);
	}

	double prodotto_finale;

	MPI_Reduce(&prodotto_parziale,&prodotto_finale,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) printf("Prodotto = %d\n", (int)prodotto_finale);





	MPI_Finalize();
	return 0;
}