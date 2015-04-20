#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_darg= dummyarg-1 dummyarg-2
OUT_host= host2 host1 
OUT_bug403= bug1OK #bug1
OUT_module= mod1

#host3 host2-1 ... problem in gfortran?

OUT=$(OUT_module) $(OUT_bug403) $(OUT_darg) $(OUT_host)

include ../Makefile.common
