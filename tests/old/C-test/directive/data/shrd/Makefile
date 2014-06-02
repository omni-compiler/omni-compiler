# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= shrd001 shrd002 shrd003 shrd004 \
		  shrd005 shrd006 shrd007 shrd008 \
		  shrd009 shrd010 shrd011 shrd012 \
		  shrd013 shrd014 shrd015 shrd016 \
		  shrd017 shrd018 shrd019 shrd020 \
		  shrd021 shrd022
OBJS		= shrd001.o shrd002.o shrd003.o shrd004.o \
		  shrd005.o shrd006.o shrd007.o shrd008.o \
		  shrd009.o shrd010.o shrd011.o shrd012.o \
		  shrd013.o shrd014.o shrd015.o shrd016.o \
		  shrd017.o shrd018.o shrd019.o shrd020.o \
		  shrd021.o shrd022.o
SRCS		= shrd001.c shrd002.c shrd003.c shrd004.c \
		  shrd005.c shrd006.c shrd007.c shrd008.c \
		  shrd009.c shrd010.c shrd011.c shrd012.c \
		  shrd013.c shrd014.c shrd015.c shrd016.c \
		  shrd017.c shrd018.c shrd019.c shrd020.c \
		  shrd021.c shrd022.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
