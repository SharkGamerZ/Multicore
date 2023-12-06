#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double genera(int a,int b) {
	double range = (b-a);
	double div = RAND_MAX / range;
	return a + (rand() / div);
}

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	int number_of_tosses = 5000000;
	double x, y, distance_squared;

	int number_in_circle = 0;
	for (int toss = 0; toss < number_of_tosses; toss++) {
		x = genera(-1,1);
		y = genera(-1,1);

		distance_squared = x*x + y*y;
		if (distance_squared <= 1) number_in_circle++;
		//printf("x:%f y:%f distance_squared:%f\n",x,y, distance_squared);
		
	}

	double pi_estimate = 4*number_in_circle/ (double) number_of_tosses;
	
	printf("%f\n",pi_estimate );
	return 0;
}


