import re
with open('ds4fa/ds4.c', 'r') as f:
    text = f.read()

# Revert
text = text.replace("for (uint32_t il = e->rpc_layer_start; il < e->rpc_layer_end; il++)", "for (uint32_t il = 0; il < DS4_N_LAYER; il++)")

with open('ds4fa/ds4.c', 'w') as f:
    f.write(text)
