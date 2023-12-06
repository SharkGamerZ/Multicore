#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>

void funzione(void);
int change_page_permissions_of_address(void *addr);

int main(void) {
    void *funzione_p = (void*)funzione;

    // Cambia i permessi della pagina che contiene funzione() per leggere, scrivere ed eseguire
    // Funziona solo se funzione() è in una singola pagina
    if(change_page_permissions_of_address(funzione_p) == -1) {
        fprintf(stderr, "Error while changing page permissions of foo(): %s\n", strerror(errno));
        return 1;
    }

    int pid = fork();

    if (pid == 0) {
    	sleep(1);
    	printf("Processo figlio a %p:",funzione_p);
    	funzione();
    }
    else {
    	// Change the immediate value in the addl instruction in foo() to 42
	    unsigned char *instruction = (unsigned char*)funzione_p + 18;
	    *instruction = 0x2;
    	printf("Processo padre a %p:",funzione_p);
    	funzione();
    	sleep(1);
    }
	

    return 0;
}

void funzione(void) {
    int i=0;
    i = i + 1;
    printf("i: %d\n", i);
}

int change_page_permissions_of_address(void *addr) {
    // Move the pointer to the page boundary
    int page_size = getpagesize();
    addr -= (unsigned long)addr % page_size;

    if(mprotect(addr, page_size, PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
        return -1;
    }

    return 0;
}