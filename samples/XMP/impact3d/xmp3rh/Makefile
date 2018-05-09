FC = xmpf90
FFLAGS = -O3 --Wf-fopenmp --Wn-fopenmp -I../share
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
	$(FC) $(FFLAGS) mod-*.o impact3d.F90  -o $(LOAD) $(LIBS)
 
# library
libi3d.a  : libi3d.a(adv3dx.o)\
            libi3d.a(adv3dy.o)\
            libi3d.a(adv3dz.o)\
            libi3d.a(cmp3dp.o)\
            libi3d.a(cmpram.o)\
            libi3d.a(init.o)
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
 
# module member
constant.mod : mod-constant.f90 
	$(FC) $(FFLAGS) -c mod-constant.f90
parameter.mod : mod-parameter.f90 
	$(FC) $(FFLAGS) -c mod-parameter.f90
phys.mod : mod-phys.f90 parameter.mod
	$(FC) $(FFLAGS) -c mod-phys.f90
 
# clrean
clean:
	rm -f *.s *.o *.mod *.xmod *.a __omni* $(LOAD)
