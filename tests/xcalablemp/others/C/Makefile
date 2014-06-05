XCC     = xmpcc -O2
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x .c

#-- default rules
.c.x:
	$(XCC) -o $@ $<

.x:
	$(XRUN) -n 4 $<


#-- special rules
278-hyphen_d.x: 278-hyphen_d.c
	$(XCC) -o $@ $< -DDEFINED_AT_COMPILE_TIME='"happy'\ guppy\"


#-- cleaning
clean:
	rm -f $(EXES) $(C_OBJS)