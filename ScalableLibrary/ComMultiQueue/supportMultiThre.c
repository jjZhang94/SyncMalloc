#include <pthread.h>

#include <stdio.h>

#include <string.h>

#include <unistd.h>

void *ThreadFunc()
{
    static int count = 1;
    printf ("Create thread %d\n", count);
    count++;
}

int main(void)
{
    int     err;
    pthread_t tid;
    while (1)
    {
           err= pthread_create(&tid, NULL, ThreadFunc, NULL);
           if(err != 0){
               printf("can't create thread: %s\n",strerror(err));
               return 1;
           break;
           }
          usleep(2000);
    }
}
