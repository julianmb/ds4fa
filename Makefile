CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=gnu99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)
HIP_SRCS := $(wildcard hip/*.hip)

BACKEND ?= auto

ifeq ($(BACKEND),auto)
ifeq ($(UNAME_S),Darwin)
BACKEND = metal
else
BACKEND = native
endif
endif

ifeq ($(BACKEND),metal)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else ifeq ($(BACKEND),rocm)
HIPCC ?= hipcc
HIP_CFLAGS = -O3 -std=c++11 -DDS4_USE_ROCM --offload-arch=gfx1151
HIP_LDLIBS = $(LDLIBS)
HIP_OBJS = hip/argsort.o hip/bin.o hip/concat.o hip/cpy.o hip/dense.o \
           hip/dsv4_hc.o hip/dsv4_kv.o hip/dsv4_misc.o hip/dsv4_rope.o \
           hip/flash_attn.o hip/get_rows.o hip/glu.o hip/moe.o hip/norm.o \
           hip/repeat.o hip/set_rows.o hip/softmax.o hip/sum_rows.o hip/unary.o \
           ds4_npu.o

# Optionally link XRT if present
XRT_LIB = $(shell if [ -d /opt/xilinx/xrt/lib ]; then echo "-L/opt/xilinx/xrt/lib -lxrt_coreutil"; else echo ""; fi)
XRT_INC = $(shell if [ -d /opt/xilinx/xrt/include ]; then echo "-I/opt/xilinx/xrt/include"; else echo ""; fi)
HIP_CFLAGS += $(XRT_INC)
HIP_LDLIBS += $(XRT_LIB)

CORE_OBJS = ds4.o ds4_hip.o $(HIP_OBJS)
NATIVE_CORE_OBJS = ds4_native.o
CFLAGS += -DDS4_USE_ROCM
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all clean test

all: ds4 ds4-server

ifeq ($(BACKEND),metal)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else ifeq ($(BACKEND),rocm)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(HIPCC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(HIP_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(HIPCC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(HIP_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
endif

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)

ds4.o: ds4.c ds4.h ds4_metal.h ds4_hip.h
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

ds4_native.o: ds4.c ds4.h ds4_metal.h ds4_hip.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_hip.o: ds4_hip.cpp ds4_hip.h
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ ds4_hip.cpp

ds4_npu.o: ds4_npu.cpp ds4_npu.h
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ ds4_npu.cpp

hip/%.o: hip/%.hip
	$(HIPCC) $(HIP_CFLAGS) -c -o $@ $<

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	ifeq ($(BACKEND),metal)
		$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
	else ifeq ($(BACKEND),rocm)
		$(HIPCC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(HIP_LDLIBS)
	else
		$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(LDLIBS)
	endif

test: ds4_test
	./ds4_test

npu-test: ds4
	./ds4 --npu-test

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test *.o hip/*.o tests/*.o
