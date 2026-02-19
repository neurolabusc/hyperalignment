UNAME := $(shell uname -s)

CC = cc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDFLAGS =

ifeq ($(UNAME),Darwin)
	LDFLAGS += -framework Accelerate
else
	LDFLAGS += -llapack -lblas -lm
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
		LDFLAGS = -framework Accelerate -fsanitize=address
	else
		LDFLAGS = -llapack -lblas -lm -fsanitize=address
	endif
endif

SRCS = ha_linalg.c ha_procrustes.c ha_ridge.c ha_sparse.c \
       ha_searchlight.c ha_template.c ha_ensemble.c
OBJS = $(SRCS:.c=.o)
LIB = libhyperalignment.a
TEST = ha_test

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST)
	./$(TEST)

$(TEST): ha_test.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lhyperalignment $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) $(LIB) $(TEST)

.PHONY: all test clean
