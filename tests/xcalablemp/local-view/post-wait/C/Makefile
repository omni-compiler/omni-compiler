XCC     = xmpcc
XRUN    = mpiexec
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

run: 
	$(XRUN) -np 4 ./post_wait.x
	$(XRUN) -np 4 ./post_wait_massive.x
	$(XRUN) -np 4 ./post_wait_massive_2.x

clean:
	rm -f $(EXES) $(C_OBJS)
