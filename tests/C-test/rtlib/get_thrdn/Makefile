# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= get_thrdn001 get_thrdn002 get_thrdn003 get_thrdn004 \
		  get_thrdn005 get_thrdn006
OBJS		= get_thrdn001.o get_thrdn002.o get_thrdn003.o get_thrdn004.o \
		  get_thrdn005.o get_thrdn006.o
SRCS		= get_thrdn001.c get_thrdn002.c get_thrdn003.c get_thrdn004.c \
		  get_thrdn005.c get_thrdn006.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
