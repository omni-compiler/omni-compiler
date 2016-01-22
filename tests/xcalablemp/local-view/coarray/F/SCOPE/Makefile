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
BUG460= sakagami460-join.x sakagami460-2.x #bug460-3.x (err in SX because of bug479)
BUG472= bug472.x bug472G1.x bug472G2.x 

SPECIALS= # bug460get.x bug460put.x

#OUT_err=entry1ERR

#host3 host2-1 ... problem in gfortran?

OUT= $(OUT_bug) $(OUT_host) $(OUT_module) $(OUT_epi) $(OUT_darg) 

include ../Makefile.common

bug460-mod.o: bug460-mod.f90
	xmpf90 -c $(OPT) $<

bug460get.x: bug460-mod.o bug460get-main.f90
	xmpf90 $(OPT) -o $@ bug460-mod.o bug460get-main.f90

bug460put.x: bug460-mod.o bug460put-main.f90
	xmpf90 $(OPT) -o $@ bug460-mod.o bug460put-main.f90


bug460-3-mod.o: bug460-3-mod.f90
	xmpf90 -c $(OPT) $<

bug460-3.x: bug460-3-mod.o bug460-3.f90
	xmpf90 $(OPT) -o $@ bug460-3-mod.o bug460-3.f90

