# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= thdprvt001 thdprvt002 thdprvt003 thdprvt004 \
		  thdprvt005 thdprvt006 thdprvt007 thdprvt008 \
		  thdprvt009 thdprvt010 thdprvt011 thdprvt012 \
		  thdprvt013 thdprvt014 thdprvt015 thdprvt016 \
		  thdprvt017 thdprvt018 thdprvt019 thdprvt020 \
		  thdprvt021 thdprvt022 thdprvt023 thdprvt024 \
		  thdprvt025 thdprvt026 thdprvt027
OBJS		= thdprvt001.o thdprvt002.o thdprvt003.o thdprvt004.o \
		  thdprvt005.o thdprvt006.o thdprvt007.o thdprvt008.o \
		  thdprvt009.o thdprvt010.o thdprvt011.o thdprvt012.o \
		  thdprvt013.o thdprvt014.o thdprvt015.o thdprvt016.o \
		  thdprvt017.o thdprvt018.o thdprvt019.o thdprvt020.o \
		  thdprvt021.o thdprvt022.o thdprvt023.o thdprvt024.o \
		  thdprvt025.o thdprvt026.o thdprvt027.o global.o
SRCS		= thdprvt001.c thdprvt002.c thdprvt003.c thdprvt004.c \
		  thdprvt005.c thdprvt006.c thdprvt007.c thdprvt008.c \
		  thdprvt009.c thdprvt010.c thdprvt011.c thdprvt012.c \
		  thdprvt013.c thdprvt014.c thdprvt015.c thdprvt016.c \
		  thdprvt017.c thdprvt018.c thdprvt019.c thdprvt020.c \
		  thdprvt021.c thdprvt022.c thdprvt023.c thdprvt024.c \
		  thdprvt025.c thdprvt026.c thdprvt027.c global.c


all:	${PROGRAMS}

thdprvt004: thdprvt004.o global.o
	$(CC) $(CFLAGS) -o thdprvt004 thdprvt004.o global.o

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
