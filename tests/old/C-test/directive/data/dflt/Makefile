# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= dflt001 dflt002 dflt003 dflt004 \
		  dflt005 dflt006 dflt007 dflt008
OBJS		= dflt001.o dflt002.o dflt003.o dflt004.o \
		  dflt005.o dflt006.o dflt007.o dflt008.o
SRCS		= dflt001.c dflt002.c dflt003.c dflt004.c \
		  dflt005.c dflt006.c dflt007.c dflt008.c

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
