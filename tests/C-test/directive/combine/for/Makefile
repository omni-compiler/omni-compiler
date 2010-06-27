PROGRAMS	= prlfor001 prlfor002 prlfor003 prlfor004 \
	          prlfor005 prlfor006 prlfor007 prlfor008 \
		  prlfor009 prlfor010 prlfor011 prlfor012
OBJS		= prlfor001.o prlfor002.o prlfor003.o prlfor004.o \
	          prlfor005.o prlfor006.o prlfor007.o prlfor008.o \
		  prlfor009.o prlfor010.o prlfor011.o prlfor012.o
SRCS		= prlfor001.c prlfor002.c prlfor003.c prlfor004.c \
	          prlfor005.c prlfor006.c prlfor007.c prlfor008.c \
		  prlfor009.c prlfor010.c prlfor011.c prlfor012.c

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
