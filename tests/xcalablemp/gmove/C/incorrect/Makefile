XCC     = xmpcc -O2
XRUN    = mpiexec
F_TESTS = $(wildcard *.c)
F_EXES  = $(F_TESTS:.c=.x)
F_OBJS  = $(F_TESTS:.c=.o)
EXES    = $(F_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

.c.x:
	$(XCC) -o $@ $<

run: $(EXES)
	$(XRUN) -n 2 ./gmove_ata_2a1t_c.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_c.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_bc.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a1t_c.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_b_NG.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b_NG.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b_NG.x

clean:
	rm -f $(EXES) $(F_OBJS)
