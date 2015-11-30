#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_darg= dummyarg-1.x dummyarg-2.x dummyarg-3.x
OUT_host= host2.x host1.x
OUT_module= mod1.x modalloc.x nestModule.x
OUT_epi= autosyncall1.x finalize.x
OUT_bug= $(BUG403) $(BUG451) $(BUG457) $(BUG458) $(BUG459) $(BUG460) $(BUG472)
BUG403= bug403.x bug403OK.x   #403 Problem FuncDefBlock.Finalize()
	                      # To avoid this problem, the current implementation
	                      # does not allow to use both CAF and XMP together.
BUG451= bug451.x
BUG457= sakagami457.x
BUG458= sakagami458-join.x 
BUG459= sakagami459-join.x 
BUG460= sakagami460-join.x sakagami460-2.x
BUG472= bug472.x bug472G1.x bug472G2.x 

#OUT_err=entry1ERR

#host3 host2-1 ... problem in gfortran?

OUT= $(OUT_bug) $(OUT_host) $(OUT_module) $(OUT_epi) $(OUT_darg) 

include ../F/Makefile.common
