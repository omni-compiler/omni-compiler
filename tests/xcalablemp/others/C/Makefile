XCC     = xmpcc -O2
XRUN    = mpirun
C_TESTS = $(wildcard *.c)
C_EXES  = $(C_TESTS:.c=.x)
C_OBJS  = $(C_TESTS:.c=.o)
EXES    = $(C_EXES)

.PHONY: clean all default run

all default: $(EXES)

.SUFFIXES: .x

.c.x:
	$(XCC) -o $@ $<

#special handling
hyphen_d.x: hyphen_d.c
	$(XCC) -o $@ $< -DDEFINED_AT_COMPILE_TIME='"happy'\ guppy\"

run: macro_in_pragma hyphen_d

macro_in_pragma:
	$(XRUN) -n 4 macro_in_pragma.x

hyphen_d:
	$(XRUN) -n 4 hyphen_d.x

clean:
	rm -f $(EXES) $(C_OBJS)