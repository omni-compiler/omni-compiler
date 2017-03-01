MAKE = make --no-print-directory
XCC = ompcc -acc
XRUN = mpirun -n 1
EXE = declare_extern.x

OBJS = declare_extern.o declare_extern_sub.o

.PHONY : all clean run

all: $(EXE)

$(EXE): $(OBJS)
	$(XCC) -o $@ $^

run: $(EXE)
	$(XRUN) ./$(EXE)

.c.o:
	$(XCC) -c $<

clean:
	rm -f $(EXE) $(OBJS)

declare_extern.o:  declare_extern.h
declare_extern_sub.o:  declare_extern.h
