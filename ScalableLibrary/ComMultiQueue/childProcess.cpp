// child_process.c
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <mqueue.h>
#include <unistd.h>
#include <string.h>

int main() {
    mqd_t mq_main;
    mqd_t mq_child;
    struct mq_attr attr;
    message msg_buffer;
    char child_queue_name[MAX_MSG_SIZE];

    snprintf(child_queue_name, MAX_MSG_SIZE, "%s%d", QUEUE_CHILD_PREFIX, getpid());

    attr.mq_flags = 0;
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = MAX_MSG_SIZE;
    attr.mq_curmsgs = 0;

    // Create child's queue for receiving responses
    mq_child = mq_open(child_queue_name, O_CREAT | O_RDONLY, 0644, &attr);
    if (mq_child == (mqd_t)-1) {
        perror("Child queue creation");
        exit(EXIT_FAILURE);
    }

    // Open main queue to send message
    mq_main = mq_open(QUEUE_MAIN, O_WRONLY);
    if (mq_main == (mqd_t)-1) {
        perror("Main queue open");
        exit(EXIT_FAILURE);
    }

    // Send message to main process
    snprintf(msg_buffer.text, MAX_MSG_SIZE, "Hello from %s", child_queue_name);
    mq_send(mq_main, (const char*)&msg_buffer, MAX_MSG_SIZE, 0);

    // Wait and read the response
    mq_receive(mq_child, (char*)&msg_buffer, MAX_MSG_SIZE, NULL);
    printf("Child received: %s\n", msg_buffer.text);

    mq_close(mq_main);
    mq_close(mq_child);
    mq_unlink(child_queue_name);

    return 0;
}
