XCC = ompcc
XCFLAGS = -acc
LDFLAGS = -acc
EXES = data.x kernel.x

.PHONY: all clean run

all: $(EXES)

run: $(EXES)
	@for exe in $(EXES); do $(XRUN) ./$$exe; done

%.x: %.o main.o
	$(XCC) -o $@ $^ $(LDFLAGS)
%.o : %.c
	$(XCC) -c $< $(XCFLAGS)
main.o : main.c
	$(CC)  -c $< $(CFLAGS)

clean:
	rm -f *.o *.x

