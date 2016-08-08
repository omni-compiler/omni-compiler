MAKE = make --no-print-directory
XCC = ompcc -acc
XRUN = mpirun -n 1
EXE = data_extern.x

OBJS = data_extern.o data_extern_sub.o

.PHONY : all clean run

all: $(EXE)

$(EXE): $(OBJS)
	$(XCC) -o $@ $^

run: $(EXE)
	$(XRUN) $(EXE)

.c.o:
	$(XCC) -c $<

clean:
	rm -f $(EXE) $(OBJS)

data_extern.o:  data_extern.h
data_extern_sub.o:  data_extern.h
