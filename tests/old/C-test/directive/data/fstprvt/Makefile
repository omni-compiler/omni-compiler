# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= fstprvt001 fstprvt002 fstprvt003 fstprvt004 \
		  fstprvt005 fstprvt006 fstprvt007 fstprvt008 \
		  fstprvt009 fstprvt010 fstprvt011 fstprvt012 \
		  fstprvt013 fstprvt014 fstprvt015 fstprvt016 \
		  fstprvt017 fstprvt018 fstprvt019 fstprvt020 \
		  fstprvt021 fstprvt022 fstprvt023 fstprvt024 \
		  fstprvt025 fstprvt026 fstprvt027 fstprvt028
OBJS		= fstprvt001.o fstprvt002.o fstprvt003.o fstprvt004.o \
		  fstprvt005.o fstprvt006.o fstprvt007.o fstprvt008.o \
		  fstprvt009.o fstprvt010.o fstprvt011.o fstprvt012.o \
		  fstprvt013.o fstprvt014.o fstprvt015.o fstprvt016.o \
		  fstprvt017.o fstprvt018.o fstprvt019.o fstprvt020.o \
		  fstprvt021.o fstprvt022.o fstprvt023.o fstprvt024.o \
		  fstprvt025.o fstprvt026.o fstprvt027.o fstprvt028.o
SRCS		= fstprvt001.c fstprvt002.c fstprvt003.c fstprvt004.c \
		  fstprvt005.c fstprvt006.c fstprvt007.c fstprvt008.c \
		  fstprvt009.c fstprvt010.c fstprvt011.c fstprvt012.c \
		  fstprvt013.c fstprvt014.c fstprvt015.c fstprvt016.c \
		  fstprvt017.c fstprvt018.c fstprvt019.c fstprvt020.c \
		  fstprvt021.c fstprvt022.c fstprvt023.c fstprvt024.c \
		  fstprvt025.c fstprvt026.c fstprvt027.c fstprvt028.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
