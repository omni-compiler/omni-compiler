# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= prvt001 prvt002 prvt003 prvt004 \
		  prvt005 prvt006 prvt007 prvt008 \
		  prvt009 prvt010 prvt011 prvt012 \
		  prvt013 prvt014 prvt015 prvt016 \
		  prvt017 prvt018 prvt019 prvt020 \
		  prvt021 prvt022 prvt023 prvt024 \
		  prvt025 prvt026
OBJS		= prvt001.o prvt002.o prvt003.o prvt004.o \
		  prvt005.o prvt006.o prvt007.o prvt008.o \
		  prvt009.o prvt010.o prvt011.o prvt012.o \
		  prvt013.o prvt014.o prvt015.o prvt016.o \
		  prvt017.o prvt018.o prvt019.o prvt020.o \
		  prvt021.o prvt022.o prvt023.o prvt024.o \
		  prvt025.o prvt026.o
SRCS		= prvt001.c prvt002.c prvt003.c prvt004.c \
		  prvt005.c prvt006.c prvt007.c prvt008.c \
		  prvt009.c prvt010.c prvt011.c prvt012.c \
		  prvt013.c prvt014.c prvt015.c prvt016.c \
		  prvt017.c prvt018.c prvt019.c prvt020.c \
		  prvt021.c prvt022.c prvt023.c prvt024.c \
		  prvt025.c prvt026.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
