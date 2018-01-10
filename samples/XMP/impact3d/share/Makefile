FC = mpif90
FFLAGS = -O3
CC = gcc
CFLAGS = -O
AR = ar
ARFLAG = -rc
RANLIB = ar
RANLIBFLAG = -ts
 
# library
libmeasure.a  : libmeasure.a(measure.o)\
                libmeasure.a(fgetwtod.o)
	@$(RANLIB) $(RANLIBFLAG) libmeasure.a 
 
libmeasure.a(measure.o) : measure.f90
	$(FC) $(FFLAGS) -c measure.f90
	@$(AR) $(ARFLAG) libmeasure.a measure.o
	@/bin/rm -f measure.o
libmeasure.a(fgetwtod.o) : fgetwtod.c
	$(CC) $(CFLAGS) -c fgetwtod.c
	@$(AR) $(ARFLAG) libmeasure.a fgetwtod.o
	@/bin/rm -f fgetwtod.o
 
clean:
	rm -f *.o  *.a
