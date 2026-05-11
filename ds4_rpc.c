#include "ds4_rpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <errno.h>

struct ds4_rpc_state {
    ds4_rpc_role role;
    int fd;
};

static void set_tcp_nodelay(int fd) {
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));
}

ds4_rpc_state *ds4_rpc_init(ds4_rpc_role role, const char *worker_ip, int port) {
    ds4_rpc_state *s = calloc(1, sizeof(ds4_rpc_state));
    s->role = role;
    s->fd = -1;

    if (role == DS4_RPC_MASTER) {
        printf("ds4_rpc: Starting as Master, connecting to Worker at %s:%d...\n", worker_ip, port);
        s->fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, worker_ip, &addr.sin_addr);

        while (connect(s->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            printf("ds4_rpc: Waiting for worker...\n");
            sleep(1);
        }
        set_tcp_nodelay(s->fd);
        printf("ds4_rpc: Connected to Worker.\n");
    } else if (role == DS4_RPC_WORKER) {
        printf("ds4_rpc: Starting as Worker, listening on port %d...\n", port);
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("bind"); exit(1);
        }
        listen(server_fd, 1);
        s->fd = accept(server_fd, NULL, NULL);
        set_tcp_nodelay(s->fd);
        close(server_fd);
        printf("ds4_rpc: Master connected.\n");
    }
    return s;
}

bool ds4_rpc_tx(ds4_rpc_state *rpc, const void *data, uint64_t bytes) {
    const uint8_t *p = data;
    while (bytes > 0) {
        ssize_t n = send(rpc->fd, p, bytes, 0);
        if (n <= 0) return false;
        p += n;
        bytes -= n;
    }
    return true;
}

bool ds4_rpc_rx(ds4_rpc_state *rpc, void *data, uint64_t bytes) {
    uint8_t *p = data;
    while (bytes > 0) {
        ssize_t n = recv(rpc->fd, p, bytes, MSG_WAITALL);
        if (n <= 0) return false;
        p += n;
        bytes -= n;
    }
    return true;
}

void ds4_rpc_close(ds4_rpc_state *rpc) {
    if (rpc) {
        if (rpc->fd >= 0) close(rpc->fd);
        free(rpc);
    }
}
