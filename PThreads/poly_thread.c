#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <pthread.h>

void funzione(void);
void processo(void);
int change_page_permissions_of_address(void *addr);

int thread_count;

int main(void) {
    void *funzione_p = (void*)funzione;

    // Cambia i permessi della pagina che contiene funzione() per leggere, scrivere ed eseguire
    // Funziona solo se funzione() è in una singola pagina
    if(change_page_permissions_of_address(funzione_p) == -1) {
        fprintf(stderr, "Error while changing page permissions of foo(): %s\n", strerror(errno));
        return 1;
    }

    printf("Processo padre:");
    funzione();

    long thread;
    pthread_t* thread_handles;

    thread_count = strtol(argv[1],NULL, 10);

    thread_handles = malloc(thread_count * sizeof(pthread_t));



    for (thread = 0; thread < thread_count; thread++) {
        printf("Processo figlio:");
        pthread_create(&thread_handles[thread], NULL, processo, *(void*));
    }

    for (thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);

    printf("Processo padre di nuovo:");
    funzione();


    return 0;
}

void funzione(void) {
    int i=0;
    i = i + 1;
    printf("i: %d\n", i);
}

void processo(void) {
    void *funzione_p = (void*)funzione;
    // Change the immediate value in the addl instruction in foo() to 42
    unsigned char *instruction = (unsigned char*)funzione_p + 18;
    *instruction = 0x2;

    funzione();
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