UNAME := $(shell uname -s)

CC = cc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS =

ifeq ($(UNAME),Darwin)
	LDFLAGS += -framework Accelerate -framework Metal \
	           -framework MetalPerformanceShaders -framework Foundation
	METAL_SRCS = ha_metal.m
	METAL_OBJS = $(METAL_SRCS:.m=.o)
	SHARED_LIB = libhyperalignment.dylib
	SHARED_FLAGS = -dynamiclib
else
	LDFLAGS += -llapack -lblas -lm
	METAL_OBJS =
	SHARED_LIB = libhyperalignment.so
	SHARED_FLAGS = -shared
endif

ifdef OPENMP
	ifeq ($(UNAME),Darwin)
		CFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/include
		LDFLAGS += -L/opt/homebrew/lib -lomp
	else
		CFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
endif

ifdef DEBUG
	CFLAGS = -std=c11 -g -O0 -Wall -Wextra -Wpedantic -fsanitize=address
	ifeq ($(UNAME),Darwin)
		LDFLAGS = -framework Accelerate -framework Metal \
		          -framework MetalPerformanceShaders -framework Foundation \
		          -fsanitize=address
	else
		LDFLAGS = -llapack -lblas -lm -fsanitize=address
	endif
endif

SRCS = ha_linalg.c ha_procrustes.c ha_ridge.c ha_sparse.c \
       ha_searchlight.c ha_template.c ha_ensemble.c
OBJS = $(SRCS:.c=.o) $(METAL_OBJS)
LIB = libhyperalignment.a
TEST = ha_test

all: $(LIB) $(SHARED_LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) $(SHARED_FLAGS) $^ $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m
	$(CC) -O2 -Wall -Wextra -c $< -o $@

test: $(TEST)
	./$(TEST)

$(TEST): ha_test.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lhyperalignment $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) $(LIB) $(SHARED_LIB) $(TEST)

.PHONY: all test clean
