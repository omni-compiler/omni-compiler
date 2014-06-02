PROGRAMS	= parallel001 parallel002 parallel003 parallel004 \
	          parallel005 parallel006 parallel007 parallel008 \
	          parallel009 parallel010 parallel011

OBJS		= parallel001.o parallel002.o parallel003.o parallel004.o \
	          parallel005.o parallel006.o parallel007.o parallel008.o \
	          parallel009.o parallel010.o parallel011.o

SRCS		= parallel001.c parallel002.c parallel003.c parallel004.c \
	          parallel005.c parallel006.c parallel007.c parallel008.c \
	          parallel009.c parallel010.c parallel011.c


all:	${PROGRAMS}

clean:
	rm -f ${PROGRAMS} ${OBJS} *~

test:	${PROGRAMS}
	@for prog in ${PROGRAMS}; do	\
	  echo "run : $$prog";		\
	  $(EXEC) ./$$prog;		\
	done
