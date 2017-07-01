
#MKL_LIB = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
#CC = icc
#CFLAGS += -O3 -I ./ -std=c99 -qopenmp -ipo -g
#LOADLIBES =  -std=c99 -lrt -lpthread -lm -qopenmp -ipo $(MKL_LIB)


LAPACK_DIR = /home/msato/misc/lapack-3.7.0
ABT_DIR = /home/msato/argobots
CFLAGS = -std=c99 -g -I$(ABT_DIR)/include 
LOADLIBES = $(LAPACK_DIR)/liblapack.a  $(LAPACK_DIR)/librefblas.a -lgfortran -lm
ABT_LIBS = $(ABT_DIR)/lib/libabt.a -lpthread
# OBJS = cholesky.o
CC = icc

cholesky_task: cholesky_task.o tasklet.o
	$(CC) -o cholesky_task cholesky_task.o tasklet.o $(ABT_LIBS) $(LOADLIBES)

cholesky: cholesky.o
	$(CC) -o cholesy cholesky.o $(LOADLIBES) 

clean:
	rm -f *.o
