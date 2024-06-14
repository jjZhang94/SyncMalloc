// main_process.c
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <mqueue.h>
#include <string.h>

int main() {
    mqd_t mq_main;
    mqd_t mq_child;
    struct mq_attr attr;
    message msg_buffer;
    char child_queue_name[MAX_MSG_SIZE];

    attr.mq_flags = 0;
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = MAX_MSG_SIZE;
    attr.mq_curmsgs = 0;

    mq_main = mq_open(QUEUE_MAIN, O_CREAT | O_RDONLY, 0644, &attr);
    if (mq_main == (mqd_t)-1) {
        perror("Main queue open");
        exit(EXIT_FAILURE);
    }

    while (1) {
        if (mq_receive(mq_main, (char*)&msg_buffer, MAX_MSG_SIZE, NULL) > 0) {
            printf("Received: %s\n", msg_buffer.text);

            // Extract child queue name from message
            sscanf(msg_buffer.text, "%s", child_queue_name);

            // Open child's queue to send response
            mq_child = mq_open(child_queue_name, O_WRONLY);
            if (mq_child == (mqd_t)-1) {
                perror("Child queue open");
                continue;
            }

            // Prepare and send response
            snprintf(msg_buffer.text, MAX_MSG_SIZE, "Response to %s", child_queue_name);
            mq_send(mq_child, (const char*)&msg_buffer, MAX_MSG_SIZE, 0);

            mq_close(mq_child);
        }
    }

    mq_close(mq_main);
    mq_unlink(QUEUE_MAIN);

    return 0;
}
