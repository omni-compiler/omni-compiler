MAKE = make --no-print-directory
XCC = ompcc -acc
XRUN = mpirun -n 1
TYPES = default char short int float double

.PHONY : all clean run

ifdef TYPE

ifeq ($(TYPE),default)
C_TESTS = $(wildcard *.c)
SUF = x
TYPEDEF = 
else
C_TESTS = $(wildcard *_any.c)
SUF = $(TYPE).x
TYPEDEF = -D`echo $(TYPE) | tr '[a-z]' '[A-Z]'`
endif

EXES = $(C_TESTS:.c=.$(SUF))
%.$(SUF) : %.c
	$(XCC) -o $*.$(SUF) $< $(TYPEDEF)

all: $(EXES)
run: $(EXES)
	@for exe in $(EXES); do $(XRUN) ./$$exe; done
clean:
	rm -f *.$(SUF) *.o

else

all:
	@for type in $(TYPES); do ($(MAKE) TYPE=$$type); done
run:
	@for type in $(TYPES); do ($(MAKE) run TYPE=$$type); done
clean:
	@for type in $(TYPES); do ($(MAKE) clean TYPE=$$type); done
endif


