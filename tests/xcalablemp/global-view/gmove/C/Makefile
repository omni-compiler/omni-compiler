ONESIDED_LIB=$(shell xmpf90 --show-env | sed -n '/ONESIDED_LIB=/p' | sed 's/ONESIDED_LIB=//' | sed "s/\"//g")

XCC     = xmpcc
XRUN    = mpiexec
TESTS = $(wildcard *.c)
EXES  = $(TESTS:.c=.x)
OBJS  = $(TESTS:.c=.o)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

chk.o: ./chk/chk.c
	$(XCC) -c $^ -o $@

CFLAG =
ifeq ($(ONESIDED_LIB),MPI3)
	CFLAG = -cpp -D_MPI3
else
	CFLAG = -cpp
endif

%.x: %.c chk.o
	  $(XCC) ${CFLAG} -o $@ $< ./chk.o

run: $(EXES)
	$(XRUN) -n 2 ./gmove_lc_2.x
	$(XRUN) -n 4 ./gmove_lc_4.x
	$(XRUN) -n 8 ./gmove_lc_8.x
	$(XRUN) -n 2 ./gmove_ata_2.x
	$(XRUN) -n 4 ./gmove_ata_4.x
	$(XRUN) -n 2 ./gmove_bca_2.x
	$(XRUN) -n 4 ./gmove_bca_4.x
	$(XRUN) -n 8 ./gmove_bca_8.x
	$(XRUN) -n 12 ./gmove_bca_12.x
	$(XRUN) -n 16 ./gmove_bca_16.x
	$(XRUN) -n 2 ./gmove_cp_2.x
	$(XRUN) -n 4 ./gmove_cp_4.x
	$(XRUN) -n 8 ./gmove_cp_8.x
	$(XRUN) -n 16 ./gmove_cp_16.x
	$(XRUN) -n 2 ./gmove_G2L_2.x
	$(XRUN) -n 4 ./gmove_G2L_4.x
	$(XRUN) -n 8 ./gmove_G2L_8.x
	$(XRUN) -n 16 ./gmove_G2L_16.x
	$(XRUN) -n 4 ./337_gmove.x
	$(XRUN) -n 2 ./417_gmove.x
	$(XRUN) -n 2 ./gmove_inout_1.x
	$(XRUN) -n 8 ./gmove_inout_3.x
	$(XRUN) -n 4 ./gmove_test_normal.x
	$(XRUN) -n 8 ./gmove_test_in.x
	$(XRUN) -n 8 ./gmove_test_out.x

clean:
	rm -f $(EXES) $(OBJS) chk.o
