FC = mpif90
FFLAGS = -O3 -fopenmp -I../share
AR = ar
ARFLAG = -rc
RANLIB = ar
RANLIBFLAG = -ts
 
LOAD = impact3d
LIBS = -L. -li3d -L../share -lmeasure
 
#-------------
# main module
#-------------
impact3d : libi3d.a impact3d.F90 
	$(FC) $(FFLAGS) impact3d.F90  -o $(LOAD) $(LIBS)
 
# library
libi3d.a  : libi3d.a(adv3dx.o)\
            libi3d.a(adv3dy.o)\
            libi3d.a(adv3dz.o)\
            libi3d.a(cmp3dp.o)\
            libi3d.a(cmpram.o)\
            libi3d.a(init.o)\
            libi3d.a(shiftx.o)\
            libi3d.a(shifty.o)\
            libi3d.a(shiftz.o)
	@$(RANLIB) $(RANLIBFLAG) libi3d.a 
 
# fortran90 member
libi3d.a(adv3dx.o) : adv3dx.F90 parameter.mod constant.mod phys.mod 
	$(FC) $(FFLAGS) -c adv3dx.F90
	@$(AR) $(ARFLAG) libi3d.a adv3dx.o
	@/bin/rm -f adv3dx.o
libi3d.a(adv3dy.o) : adv3dy.F90 parameter.mod constant.mod phys.mod 
	$(FC) $(FFLAGS) -c adv3dy.F90
	@$(AR) $(ARFLAG) libi3d.a adv3dy.o
	@/bin/rm -f adv3dy.o
libi3d.a(adv3dz.o) : adv3dz.F90 parameter.mod constant.mod phys.mod 
	$(FC) $(FFLAGS) -c adv3dz.F90
	@$(AR) $(ARFLAG) libi3d.a adv3dz.o
	@/bin/rm -f adv3dz.o
libi3d.a(cmp3dp.o) : cmp3dp.F90 parameter.mod constant.mod phys.mod
	$(FC) $(FFLAGS) -c cmp3dp.F90
	@$(AR) $(ARFLAG) libi3d.a cmp3dp.o
	@/bin/rm -f cmp3dp.o
libi3d.a(cmpram.o) : cmpram.F90 parameter.mod constant.mod phys.mod
	$(FC) $(FFLAGS) -c cmpram.F90
	@$(AR) $(ARFLAG) libi3d.a cmpram.o
	@/bin/rm -f cmpram.o
libi3d.a(init.o) : init.F90 parameter.mod constant.mod phys.mod
	$(FC) $(FFLAGS) -c init.F90
	@$(AR) $(ARFLAG) libi3d.a init.o
	@/bin/rm -f init.o
libi3d.a(shiftx.o) : shiftx.f90 parameter.mod constant.mod
	$(FC) $(FFLAGS) -c shiftx.f90
	@$(AR) $(ARFLAG) libi3d.a shiftx.o
	@/bin/rm -f shiftx.o
libi3d.a(shifty.o) : shifty.f90 parameter.mod constant.mod
	$(FC) $(FFLAGS) -c shifty.f90
	@$(AR) $(ARFLAG) libi3d.a shifty.o
	@/bin/rm -f shifty.o
libi3d.a(shiftz.o) : shiftz.f90 parameter.mod constant.mod
	$(FC) $(FFLAGS) -c shiftz.f90
	@$(AR) $(ARFLAG) libi3d.a shiftz.o
	@/bin/rm -f shiftz.o
 
# module member
constant.mod : mod-constant.f90 parameter.mod
	$(FC) $(FFLAGS) -c mod-constant.f90
	@$(AR) $(ARFLAG) libi3d.a mod-constant.o
	@/bin/rm -f mod-constant.o
parameter.mod : mod-parameter.f90 
	$(FC) $(FFLAGS) -c mod-parameter.f90
	@$(AR) $(ARFLAG) libi3d.a mod-parameter.o
	@/bin/rm -f mod-parameter.o
phys.mod : mod-phys.f90 parameter.mod
	$(FC) $(FFLAGS) -c mod-phys.f90
	@$(AR) $(ARFLAG) libi3d.a mod-phys.o
	@/bin/rm -f mod-phys.o
 
# clrean
clean:
	rm -f *.o *.mod *.a $(LOAD)
