#OPT= --debug
#OPT= --Wx-fcoarray=4 --debug
#OPT= -g --debug --verbose
NPE=3

OUT_OK    = a1d1-3.x a1d1-2.x a1d1-1.x a6d3.x a6d3-1.x a6d3-2.x a3d3.x a2d2.x a2d2-1-1.x a2d2-1-5.x s-2.x s-1.x a1d1-4.x a0d0-2.x a0d0-3.x block_a0d0-2.x

OUT_NG    = a2d0.x a2d1.x a2d1-2.x a2d1-3.x
OUT_bound = #l0-3okB.x cannot run on SX
#OUT_type  = a1d1-typeNG
#OUT_err  = l0-2NG
OUT_BUG   = bug462.x
OUT_ATOMIC = atomic0.x atomic.x atomic2.x #atomicRefRemote.x
OUT_CONSIST= consistency1.x consistency2.x

OUT_NGB  = a2d2-2NGB.x

#OUT=$(OUT_OK) $(OUT_NG) $(OUT_bound) $(OUT_type) $(OUT_NGB)
OUT= $(OUT_CONSIST) $(OUT_ATOMIC) $(OUT_BUG) $(OUT_OK) $(OUT_NG) $(OUT_bound) $(OUT_NGB)

include ../Makefile.common
