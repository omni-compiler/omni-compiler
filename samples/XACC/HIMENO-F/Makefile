OBJS = himeno.o
FC = xmpf90 
NDX ?= 1
NDY ?= 1
PROG = himenoXACC_$(NDX)x$(NDY)

FFLAGS = -O2 -xacc --debug --verbose  -lstdc++ --Wn"-acc -Minfo=accel" --Wl"-acc"

LDFLAGS = ${FFLAGS}


all : ${PROG}

${PROG} : ${OBJS}
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ himeno.f90 -DNDX=$(NDX) -DNDY=$(NDY) -cpp

${OBJS} : himeno.f90
	${FC} -c ${FFLAGS} $<

clean :
	rm -f ${PROG} ${OBJS} core *~ *.xmod
