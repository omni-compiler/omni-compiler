XFC     = xmpf90 -O2
XRUN    = mpiexec
TESTS = $(wildcard *.f90)
EXES  = $(TESTS:.f90=.x)
OBJS  = $(TESTS:.f90=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .f90

chk.o: ./chk/chk.f90
	$(XFC) -c $^ -o $@

%.x: %.f90 chk.o
	$(XFC) -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 2 ./gmove_lc_1a1t_b.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_bc.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_b_off.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_bc_off.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c_off.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_b_h.x
	$(XRUN) -n 2 ./gmove_lc_1a1t_c_h.x
	$(XRUN) -n 4 ./gmove_lc_2a12t_b_gb.x
	$(XRUN) -n 4 ./gmove_lc_2a12t_b_gb2.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_bc.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_b_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_c_h.x
	$(XRUN) -n 4 ./gmove_lc_2a2t_gb.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_bc.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_b_h.x
	$(XRUN) -n 8 ./gmove_lc_3a3t_c_h.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_b_h.x
	$(XRUN) -n 2 ./gmove_ata_2a1t_c.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b2.x
	$(XRUN) -n 2 ./gmove_ata_3a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_4a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_5a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_6a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_7a1t_b_c.x
	$(XRUN) -n 2 ./gmove_ata_7a1t_b.x
	$(XRUN) -n 2 ./gmove_ata_7a1t_c.x
	$(XRUN) -n 4 ./gmove_bca_2a2t_b.x
	$(XRUN) -n 8 ./gmove_bca_3a3t_bc2.x
	$(XRUN) -n 8 ./gmove_bca_3a3t_bc.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_b2.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_b2_s.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_gb.x
	$(XRUN) -n 12 ./gmove_bca_3a3t_c_gb_s.x
	$(XRUN) -n 16 ./gmove_bca_4a4t_bc.x
	$(XRUN) -n 16 ./gmove_bca_4a4t_b2.x
	$(XRUN) -n 2 ./gmove_G2L_1a1t_b.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_bc.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_bc_b_h.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_b.x
	$(XRUN) -n 2 ./gmove_cp_1a1t_c.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_bc.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b_c.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_b.x
	$(XRUN) -n 4 ./gmove_cp_2a2t_c.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_bc.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b_c.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_b.x
	$(XRUN) -n 8 ./gmove_cp_3a3t_c.x
	$(XRUN) -n 16 ./gmove_cp_4a4t_b_c.x
	$(XRUN) -n 2 ./gmove_G2L_1a1t_bc.x
	$(XRUN) -n 4 ./gmove_G2L_2a2t_bc.x
	$(XRUN) -n 8 ./gmove_G2L_3a3t_bc.x
	$(XRUN) -n 16 ./gmove_G2L_4a4t_bc.x
	$(XRUN) -n 4 ./gmove_G2L_1a2t_bc.x
	$(XRUN) -n 8 ./gmove_G2L_1a3t_bc2.x
	$(XRUN) -n 8 ./gmove_G2L_1a3t_bc.x
	$(XRUN) -n 16 ./gmove_G2L_1a4t_bc.x
	$(XRUN) -n 4 ./gmove_G2L_index.x

clean:
	rm -f $(EXES) $(OBJS) _xmpf_module_INIT.o chk.o
