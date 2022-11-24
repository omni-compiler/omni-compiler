#
OMNI_HOME = $(shell xmpcc --show-env | grep OMNI_HOME | sed -e 's/OMNI_HOME=//' -e 's/"//g')
FC     = mpif90
FFLAGS = -I$(OMNI_HOME)/include -J$(OMNI_HOME)/include -fopenmp
LIBS = $(shell xmpf90 --show-env | grep OMNI_LINKER_OPT | sed -e 's/OMNI_LINKER_OPT=//' -e 's/"//g')

all: xmp_api_laplace xmp_api_laplace_1d

xmp_api_laplace: xmp_api_laplace.o
	$(FC) -o xmp_api_laplace xmp_api_laplace.o $(LIBS) -lxmpf

xmp_api_laplace.o: xmp_api_laplace.f90
	$(FC) $(FFLAGS) -c xmp_api_laplace.f90 

xmp_api_laplace_1d: xmp_api_laplace_1d.o
	$(FC) -o xmp_api_laplace_1d xmp_api_laplace_1d.o $(LIBS) -lxmpf

xmp_api_laplace_1d.o: xmp_api_laplace_1d.f90
	$(FC) $(FFLAGS) -c xmp_api_laplace_1d.f90
