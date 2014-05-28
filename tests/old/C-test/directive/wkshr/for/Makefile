# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= for001 for002 for003 for004 \
	          for005 for006 for007 for008 \
		  for009 for010 for011 for012
OBJS		= for001.o for002.o for003.o for004.o \
	          for005.o for006.o for007.o for008.o \
		  for009.o for010.o for011.o for012.o
SRCS		= for001.c for002.c for003.c for004.c \
	          for005.c for006.c for007.c for008.c \
		  for009.c for010.c for011.c for012.c

all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
