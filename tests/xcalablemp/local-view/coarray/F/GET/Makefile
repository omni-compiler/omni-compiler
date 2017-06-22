#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_basic  = a1d1-4.x a6d3-1.x l4.x
OUT_io     = #cn1d1-writeNGB.x (comment out for SX)
OUT_arg    = actualarg.x triplet.x actualarg-2.x actualarg-3.x
OUT_nest   = nest.x putget-4.x putget-5.x
OUT_lib    = a1d1-5.x
OUT_mod    = module-1.x module-2.x
OUT_entire = axd0-1.x axd1-1.x
OUT_bound  = cn1-2okB.x  #l0-1okB.x #cannot run on SX
OUT_struct = #str0-1.x NOW!!!
OUT_atomic = atomic_ref-i4.x atomic_ref-l4.x

OUT_NGB = cn1-1NGB.x cn2-1NGB.x putget-2NGB.x putget-3NGB.x putget-6NGB.x #  nest-2NGB putgetNGB

OUT_bug354 = cn0d0-1.x cn0d0-write.x # see bug354.f90

## error with option -O1
OUT_optO1 =  a1d1-1.x axd2-1.x
# and ../STRUCT includes some errors with -O1

#OUT=nest nest-2
OUT = $(OUT_basic) $(OUT_struct) $(OUT_io) $(OUT_arg) $(OUT_nest) $(OUT_lib) $(OUT_mod) $(OUT_entire) $(OUT_bound) $(OUT_atomic) $(OUT_bug354) $(OUT_NGB) $(OUT_optO1)

include ../Makefile.common
