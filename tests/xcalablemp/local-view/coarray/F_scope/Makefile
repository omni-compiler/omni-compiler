#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_darg= dummyarg-1.x dummyarg-2.x dummyarg-3.x
OUT_host= host2.x host1.x
OUT_bug403= bug403.x bug403OK.x #bug403 FuncDefBlock.Finalize()
OUT_module= mod1.x modalloc.x nestModule.x
OUT_epi= #autosyncall1 autosyncall2
OUT_bug= sakagami457.x sakagami458-join.x sakagami459-join.x sakagami460-join.x sakagami460-2.x


#OUT_err=entry1ERR

#host3 host2-1 ... problem in gfortran?

OUT= $(OUT_bug) $(OUT_host) $(OUT_module) $(OUT_bug403) $(OUT_epi) $(OUT_darg) 

include ../F/Makefile.common
