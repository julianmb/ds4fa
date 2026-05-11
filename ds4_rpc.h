#ifndef DS4_RPC_H
#define DS4_RPC_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Role enumeration
typedef enum {
    DS4_RPC_NONE = 0,
    DS4_RPC_MASTER, // Node 0 (Layers 0..30)
    DS4_RPC_WORKER  // Node 1 (Layers 31..61)
} ds4_rpc_role;

// Opaque state
typedef struct ds4_rpc_state ds4_rpc_state;

// Initialize as master or worker.
// Master connects to worker_ip:port. Worker listens on port.
ds4_rpc_state *ds4_rpc_init(ds4_rpc_role role, const char *worker_ip, int port);

// Send/Recv bytes over the blocking TCP_NODELAY socket
bool ds4_rpc_tx(ds4_rpc_state *rpc, const void *data, uint64_t bytes);
bool ds4_rpc_rx(ds4_rpc_state *rpc, void *data, uint64_t bytes);

void ds4_rpc_close(ds4_rpc_state *rpc);

#ifdef __cplusplus
}
#endif

#endif
