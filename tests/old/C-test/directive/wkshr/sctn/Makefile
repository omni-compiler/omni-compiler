# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= sctn001 sctn002 sctn003 sctn004 \
		  sctn005 sctn006 sctn007
OBJS		= sctn001.o sctn002.o sctn003.o sctn004.o \
		  sctn005.o sctn006.o sctn007.o
SRCS		= sctn001.c sctn002.c sctn003.c sctn004.c \
		  sctn005.c sctn006.c sctn007.c
		  

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
