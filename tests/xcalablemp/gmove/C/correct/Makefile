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
	$(XRUN) -n 2 ./gmove_lc_1a1t_b.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_b_h.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c_h.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b_h.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c_h.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b_h.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b_h.x
	$(XRUN) -n 2 ./gmove_bca_1a1t_b.x
	$(XRUN) -n 2 ./gmove_bca_1a1t_c.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_b.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_c.x
	$(XRUN) -n 4 ./gmove_cp_2a1t_b.x
	$(XRUN) -n 4 ./gmove_cp_2a1t_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_c.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_c.x

clean:
	rm -f $(EXES) $(F_OBJS)
