#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_basic  = a1d1-4.x a1d1-1.x a6d3-1.x cn1-1NGB.x cn2-1NGB.x
OUT_io     = #cn1d1-writeNGB.x (comment out for SX)
OUT_arg    = actualarg.x triplet.x actualarg-2.x actualarg-3.x
OUT_nest   = nest.x putget-2NGB.x putget-3NGB.x putget-4.x putget-5.x putget-6NGB.x #  nest-2NGB putgetNGB
OUT_lib    = a1d1-5.x
OUT_mod    = module-1.x module-2.x
OUT_entire = axd0-1.x axd1-1.x axd2-1.x
OUT_bound  = cn1-2okB.x  #l0-1okB.x #cannot run on SX
OUT_struct = #str0-1.x NOW!!!

#not supported
#OUT_struct = struct0d
#OUT=bug3

#bug #354        cnd0-1 cn0d0-write (see bug2.f90)

#OUT=nest nest-2
OUT = $(OUT_basic) $(OUT_struct) $(OUT_io) $(OUT_arg) $(OUT_nest) $(OUT_lib) $(OUT_mod) $(OUT_entire) $(OUT_bound)

include ../Makefile.common
