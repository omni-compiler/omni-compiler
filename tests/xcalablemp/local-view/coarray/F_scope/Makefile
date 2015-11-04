#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_darg= dummyarg-1.x dummyarg-2.x dummyarg-3.x
OUT_host= host2.x host1.x
OUT_bug403= bug1OK.x #bug1
OUT_module= mod1.x modalloc.x
OUT_epi= #autosyncall1 autosyncall2

#OUT_err=entry1ERR

#host3 host2-1 ... problem in gfortran?

OUT= $(OUT_host) $(OUT_module) $(OUT_bug403) $(OUT_epi) $(OUT_darg) 

include ../F/Makefile.common
