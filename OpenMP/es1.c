#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char const *argv[])
{
	
	int x,i,j = 0;
	int n = 10;

	for(x = 0; x < n; x++) {
		printf("%f\n",(double) rand() / ( RAND_MAX / 2) - 1);
	}

	
	return 0;
}