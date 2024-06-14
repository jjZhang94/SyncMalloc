#ifndef COMMON_H
#define COMMON_H

#define QUEUE_MAIN "/queue_main"
#define QUEUE_CHILD_PREFIX "/queue_child_"
#define MAX_MSG_SIZE 256

typedef struct {
    char text[MAX_MSG_SIZE];
} message;

#endif