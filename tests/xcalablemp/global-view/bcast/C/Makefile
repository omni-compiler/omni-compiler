XCC     = xmpcc
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
	$(XRUN) -n 4 ./bcast.x
	$(XRUN) -n 4 ./bcast_from_p.x
	$(XRUN) -n 4 ./bcast_from_p_on_p.x
	$(XRUN) -n 4 ./bcast_on_p.x
	$(XRUN) -n 4 ./bcast_on_split_p.x
	$(XRUN) -n 4 ./bcast_from_p_on_t2.x
	$(XRUN) -n 4 ./bcast_on_split_p_from_p.x
	$(XRUN) -n 4 ./bcast_on_t2.x
	$(XRUN) -n 8 ./bcast2.x
	$(XRUN) -n 8 ./bcast_from_p2.x
	$(XRUN) -n 8 ./bcast_on_p2.x
	$(XRUN) -n 8 ./bcast_on_split_p2.x
	$(XRUN) -n 8 ./bcast_from_p2_on_p2.x
	$(XRUN) -n 16 ./bcast_on_split_p2_from_p2.x

clean:
	rm -f $(EXES) $(OBJS)
