# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= ordrd001 ordrd002 ordrd003 ordrd004 \
		  ordrd005
OBJS		= ordrd001.o ordrd002.o ordrd003.o ordrd004.o \
		  ordrd005.o
SRCS		= ordrd001.c ordrd002.c ordrd003.c ordrd004.o \
		  ordrd005.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
