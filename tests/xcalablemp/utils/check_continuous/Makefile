CC=gcc
OPT=-std=c99 -O2

all: 1dim.x 2dims.x 3dims.x 4dims.x

1dim.x: utils.c 1dim.c
	$(CC) $(OPT) 1dim.c utils.c -o 1dim.x

2dims.x: utils.c 2dims.c
	$(CC) $(OPT) 2dims.c utils.c -o 2dims.x

3dims.x: utils.c 3dims.c
	$(CC) $(OPT) 3dims.c utils.c -o 3dims.x

4dims.x: utils.c 4dims.c
	$(CC) $(OPT) 4dims.c utils.c -o 4dims.x

check: 1dim-check 2dims-check 3dims-check 4dims-check

1dim-check:
	$(CC) $(OPT) -D_CHECK 1dim.c utils.c -o 1dim.x
	./1dim.x | sort | uniq > log-1
	$(CC) $(OPT) 1dim.c utils.c -o 1dim.x
	./1dim.x | sort | uniq > log-2
	wc log-1 log-2
	diff log-1 log-2

2dims-check:
	$(CC) $(OPT) -D_CHECK 2dims.c utils.c -o 2dims.x
	./2dims.x | sort | uniq > log-1
	$(CC) $(OPT) 2dims.c utils.c -o 2dims.x
	./2dims.x | sort | uniq > log-2
	wc log-1 log-2
	diff log-1 log-2

3dims-check:
	$(CC) $(OPT) -D_CHECK 3dims.c utils.c -o 3dims.x
	./3dims.x | sort | uniq > log-1
	$(CC) $(OPT) 3dims.c utils.c -o 3dims.x
	./3dims.x | sort | uniq > log-2
	wc log-1 log-2
	diff log-1 log-2

4dims-check:
	$(CC) $(OPT) -D_CHECK 4dims.c utils.c -o 4dims.x
	./4dims.x | sort | uniq > log-1
	$(CC) $(OPT) 4dims.c utils.c -o 4dims.x
	./4dims.x | sort | uniq > log-2
	wc log-1 log-2
	diff log-1 log-2

clean:
	rm -f *.o *.x *~ log-1 log-2 


