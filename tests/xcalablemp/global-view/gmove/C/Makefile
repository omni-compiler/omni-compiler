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
	$(XRUN) -n 2 ./gmove_lc_1a1t_bc.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_b_h.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_bc.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_gb.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_bc.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b_h.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c_h.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b_h.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_c.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b_h.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_c.x
	$(XRUN) -n 2 ./gmove_ata_4a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_5a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_6a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_7a1t_b_c.x
	$(XRUN) -n 8 ./gmove_bca_3a3t_b.x
	$(XRUN) -n 8 ./gmove_bca_3a3t_b_s.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_b2.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_gb.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_gb_s.x
	$(XRUN) -n 16 ./gmove_bca_4a4t_b.x
	$(XRUN) -n 16 ./gmove_bca_4a4t_b2.x
	$(XRUN) -n 32 ./gmove_bca_5a5t_b.x
	$(XRUN) -n 32 ./gmove_bca_5a5t_b2.x
	$(XRUN) -n 64 ./gmove_bca_6a6t_b.x
	$(XRUN) -n 64 ./gmove_bca_6a6t_b2.x
	$(XRUN) -n 128 ./gmove_bca_7a7t_b.x
	$(XRUN) -n 128 ./gmove_bca_7a7t_b2.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_b.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_b_c.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_bc.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_c.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_c.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b_bc.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_bc.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_c.x
	$(XRUN) -n 16 ./gmove_cp_4a4t_b_bc.x
	$(XRUN) -n 32 ./gmove_cp_5a5t_b_bc.x
	$(XRUN) -n 64 ./gmove_cp_6a6t_b_bc.x
	$(XRUN) -n 128 ./gmove_cp_7a7t_b_bc.x
	$(XRUN) -n 2 ./gmove_G2L_1a1t_bc.x
	$(XRUN) -n 4 ./gmove_G2L_2a2t_bc.x
	$(XRUN) -n 8 ./gmove_G2L_3a3t_bc.x
	$(XRUN) -n 16 ./gmove_G2L_4a4t_bc.x
	$(XRUN) -n 32 ./gmove_G2L_5a5t_bc.x
	$(XRUN) -n 64 ./gmove_G2L_6a6t_bc.x
	$(XRUN) -n 128 ./gmove_G2L_7a7t_bc.x
	$(XRUN) -n 4 ./gmove_G2L_1a2t_bc.x
	$(XRUN) -n 8 ./gmove_G2L_1a3t_bc.x
	$(XRUN) -n 16 ./gmove_G2L_1a4t_bc.x
	$(XRUN) -n 32 ./gmove_G2L_1a5t_bc.x
	$(XRUN) -n 64 ./gmove_G2L_1a6t_bc.x
	$(XRUN) -n 128 ./gmove_G2L_1a7t_bc.x

clean:
	rm -f $(EXES) $(F_OBJS)
