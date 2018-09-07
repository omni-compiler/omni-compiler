SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

OBJS = cg.o ${COMMON}/c_print_results.o  \
       ${COMMON}/c_randdp.o ${COMMON}/c_timers.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB}

cg.o:	cg.c mpinpb.h npbparams.h timing.h
	${CCOMPILE} cg.c

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core



