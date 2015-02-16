XCC     = xmpcc -xacc -O2
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	qsub ./reflect_acc_1.sh
	qsub ./reflect_acc_2.sh
	qsub ./reflect_acc_1_massive.sh
	qsub ./reflect_acc_2_massive.sh
	qsub ./reflect_acc_2_dup_1.sh
	qsub ./reflect_acc_2_dup_2.sh
	qsub ./parallel.sh

clean:
	rm -f $(EXES) $(OBJS)
