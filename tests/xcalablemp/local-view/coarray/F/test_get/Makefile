#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_basic  = a1d1-4 a1d1-1 a6d3-1 cn1-1NGB
OUT_io     = cn1d1-writeNGB
OUT_arg    = actualarg triplet actualarg-2 actualarg-3
OUT_nest   = nest putget-2NGB putget-3NGB putget-4 putget-5 putget-6NGB #  nest-2NGB putgetNGB
OUT_lib    = a1d1-5
###OUT_mod    = module-1 module-2
OUT_entire = axd0-1 axd1-1 axd2-1
OUT_bound  = cn1-2okB l0-1okB

#not supported
#OUT_struct = struct0d
#OUT=bug3

#bug #354        cnd0-1 cn0d0-write (see bug2.f90)

#OUT=nest nest-2
OUT = $(OUT_basic) $(OUT_io) $(OUT_arg) $(OUT_nest) $(OUT_lib) $(OUT_mod) $(OUT_entire) $(OUT_bound)

include ../Makefile.common
