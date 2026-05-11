CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=gnu99

LDLIBS ?= -lm -pthread
HIPCC ?= hipcc
HIP_CFLAGS = -O3 -std=c++11 -DDS4_USE_ROCM --offload-arch=gfx1151
HIP_LDLIBS = $(LDLIBS)
HIP_OBJS = hip/argsort.o hip/bin.o hip/concat.o hip/cpy.o hip/dense.o \
           hip/dsv4_hc.o hip/dsv4_kv.o hip/dsv4_misc.o hip/dsv4_rope.o \
           hip/flash_attn.o hip/get_rows.o hip/glu.o hip/moe.o hip/norm.o \
           hip/repeat.o hip/set_rows.o hip/softmax.o hip/sum_rows.o hip/unary.o \
           ds4_npu.o ds4_rpc.o

CORE_OBJS = ds4.o ds4_hip.o $(HIP_OBJS)
CFLAGS += -DDS4_USE_ROCM

.PHONY: all clean test

all: ds4 ds4-server

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(HIPCC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(HIP_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(HIPCC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(HIP_LDLIBS)

ds4.o: ds4.c ds4.h ds4_hip.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_hip.o: ds4_hip.cpp ds4_hip.h
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ ds4_hip.cpp

ds4_npu.o: ds4_npu.cpp ds4_npu.h
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ ds4_npu.cpp

ds4_rpc.o: ds4_rpc.c ds4_rpc.h
	$(CC) $(CFLAGS) -c -o $@ ds4_rpc.c

hip/%.o: hip/%.hip
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ $<

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(HIPCC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(HIP_LDLIBS)

test: ds4_test
	./ds4_test

npu-test: ds4
	./ds4 --npu-test

clean:
	rm -f ds4 ds4-server ds4_server_test ds4_test *.o hip/*.o tests/*.o