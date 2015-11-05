TARGET = T_Module

CC = gcc
INC = -I./include

DOPT = -DIN_GCC_FRONTEND -DIN_GCC -DHAVE_CONFIG_H -D_ZLIB_ 
WOPT = -W -Wall -Wwrite-strings -Wcast-qual -Wmissing-format-attribute -pedantic -Wno-long-long -Wno-variadic-macros -Wno-overlength-strings -w
CFLAGS = -O -fno-exceptions -fasynchronous-unwind-tables $(DOPT) $(WOPT)
LDFLAGS= -lgmp -lmpfr -lz
cur_dir=`pwd`

.SUFFIXES:.c .o .h

SRC=$(shell ls *.c)
HED=$(shell ls include/*.h)
OBJ=$(SRC:.c=.o)

all:$(TARGET)
	@test -f xmpmod || sed -e "s|@@XMP_PATH@@|$(cur_dir)|g" < utils/xmpmod.bs > xmpmod ;\
	 chmod 755 xmpmod

$(TARGET):$(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) -c $< $(INC)

#depend
xmalloc.o	:xmalloc.c       ${HED}
safe-ctype.o	:safe-ctype.c    ${HED}
import_module.o	:import_module.c ${HED}
export_module.o	:export_module.c ${HED}
trans_module.o	:trans_module.c  ${HED}

clean:
	/bin/rm -f $(OBJ)

cleanall:
	/bin/rm -f $(TARGET) $(OBJ) xmpmod
