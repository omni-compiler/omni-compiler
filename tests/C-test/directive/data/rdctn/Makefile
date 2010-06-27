# $Id$
# $TSUKUBA_Release: Omni OpenMP Compiler 3 $
# $TSUKUBA_Copyright:
#  PLEASE DESCRIBE LICENSE AGREEMENT HERE
#  $
SHELL		= /bin/sh
EXEC		= /bin/sh -c
CC		= omcc
CFLAGS		= -O

PROGRAMS	= rdctn001 rdctn002 rdctn003 rdctn004 \
		  rdctn005 rdctn006 rdctn007 rdctn008 \
		  rdctn009 rdctn010 rdctn011 rdctn012 \
		  rdctn013 rdctn014 rdctn015 rdctn016 \
		  rdctn017 rdctn018 rdctn019 rdctn020 \
		  rdctn021 rdctn022 rdctn023 rdctn024 \
		  rdctn025 rdctn026 rdctn027
OBJS		= rdctn001.o rdctn002.o rdctn003.o rdctn004.o \
		  rdctn005.o rdctn006.o rdctn007.o rdctn008.o \
		  rdctn009.o rdctn010.o rdctn011.o rdctn012.o \
		  rdctn013.o rdctn014.o rdctn015.o rdctn016.o \
		  rdctn017.o rdctn018.o rdctn019.o rdctn020.o \
		  rdctn021.o rdctn022.o rdctn023.o rdctn024.o \
		  rdctn025.o rdctn026.o rdctn027.o
SRCS		= rdctn001.c rdctn002.c rdctn003.c rdctn004.c \
		  rdctn005.c rdctn006.c rdctn007.c rdctn008.c \
		  rdctn009.c rdctn010.c rdctn011.c rdctn012.c \
		  rdctn013.c rdctn014.c rdctn015.c rdctn016.c \
		  rdctn017.c rdctn018.c rdctn019.c rdctn020.c \
		  rdctn021.c rdctn022.c rdctn023.c rdctn024.c \
		  rdctn025.c rdctn026.c rdctn027.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
