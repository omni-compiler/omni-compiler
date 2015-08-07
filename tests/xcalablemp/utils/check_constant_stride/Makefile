CC=gcc
OPT=-std=c99

all: 1dim-check 2dims-check 3dims-check 4dims-check 5dims-check 6dims-check 7dims-check

1dim-check:
	$(CC) $(OPT) -D_HEAVY 1dim.c utils.c -o 1dim.x;./1dim.x | sort | uniq > log-1
	$(CC) $(OPT) 1dim.c utils.c -o 1dim.x; ./1dim.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

2dims-check:
	$(CC) $(OPT) -D_HEAVY 2dims.c utils.c -o 2dims.x; ./2dims.x | sort | uniq > log-1
	$(CC) $(OPT) 2dims.c utils.c -o 2dims.x; ./2dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

3dims-check:
	$(CC) $(OPT) -D_HEAVY 3dims.c utils.c -o 3dims.x; ./3dims.x | sort | uniq > log-1
	$(CC) $(OPT) 3dims.c utils.c -o 3dims.x; ./3dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

4dims-check:
	$(CC) $(OPT) -D_HEAVY 4dims.c utils.c -o 4dims.x; ./4dims.x | sort | uniq > log-1
	$(CC) $(OPT) 4dims.c utils.c -o 4dims.x; ./4dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

5dims-check:
	$(CC) $(OPT) -D_HEAVY 5dims.c utils.c -o 5dims.x; ./5dims.x | sort | uniq > log-1
	$(CC) $(OPT) 5dims.c utils.c -o 5dims.x; ./5dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

6dims-check:
	$(CC) $(OPT) -D_HEAVY 6dims.c utils.c -o 6dims.x; ./6dims.x | sort | uniq > log-1
	$(CC) $(OPT) 6dims.c utils.c -o 6dims.x; ./6dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

7dims-check:
	$(CC) $(OPT) -D_HEAVY 7dims.c utils.c -o 7dims.x; ./7dims.x | sort | uniq > log-1
	$(CC) $(OPT) 7dims.c utils.c -o 7dims.x; ./7dims.x | sort | uniq > log-2
	wc -l log-1 log-2; diff log-1 log-2

clean:
	rm -f *.o *.x *~ log-1 log-2 


