#
#
#

LIBXMP_INCLUDE_DIR = ../../libxmp/include
LIBXMP_SRC_DIR = ../../libxmp/src

CC = mpicc
CFLAGS = -I$(LIBXMP_INCLUDE_DIR) -I$(LIBXMP_SRC_DIR) -I../include -std=c99 -g -Wpointer-arith

LIB_OBJS = xmpf_array.o xmpf_gcomm.o xmpf_gmove.o xmpf_index.o xmpf_misc.o \
	xmpf_loop.o xmpf_nodes.o xmpf_template.o

LIBXMP_F = libxmpf.a

target: $(LIBXMP_F) xmpf_main.o

$(LIBXMP_F): $(LIB_OBJS)
	xiar cru $(LIBXMP_F) $(LIB_OBJS)
	ranlib $(LIBXMP_F)

xmpf_main.o: xmpf_main.f
	mpif90 -c xmpf_main.f

xmpf_array.o: xmpf_array.c
xmpf_gcomm.o: xmpf_gcomm.c
xmpf_gmove.o: xmpf_gmove.c
xmpf_index.o: xmpf_index.c
xmpf_misc.o: xmpf_misc.c
xmpf_loop.o: xmpf_loop.c
xmpf_nodes.o: xmpf_nodes.c
xmpf_template.o: xmpf_template.c

.c.o:
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f *.o *~ $(LIBXMP_F)