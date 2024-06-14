#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Define a structure for thread arguments
typedef struct {
    int data;
    volatile int should_exit; // Flag to indicate the thread should exit
} thread_args;

// Thread function
void* thread_func(void* arg) {
    thread_args* args = (thread_args*)arg;
    int counter = 0;

    while (!args->should_exit) {
        printf("Thread running with argument: %d, counter: %d\n", args->data, counter++);
        sleep(1); // Simulate work by sleeping for 1 second
    }

    printf("Thread exiting...\n");
    pthread_exit(NULL);
}

int main() {
    pthread_t thread_id;
    thread_args args = { .data = 42, .should_exit = 0 }; // Initialize thread arguments

    // Create the thread
    if (pthread_create(&thread_id, NULL, thread_func, &args)) {
        fprintf(stderr, "Error creating thread\n");
        return 1;
    }

    // Let the thread run for 10 seconds
    sleep(10);

    // Signal the thread to exit
    args.should_exit = 1;

    // Wait for the thread to finish
    if (pthread_join(thread_id, NULL)) {
        fprintf(stderr, "Error joining thread\n");
        return 2;
    }

    printf("Thread successfully exited.\n");
    return 0;
}
