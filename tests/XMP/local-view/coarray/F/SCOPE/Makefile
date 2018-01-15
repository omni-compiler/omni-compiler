#OPT= -g --debug
#OPT= -g --debug --verbose
NPE=3

OUT_darg= dummyarg-1.x dummyarg-2.x dummyarg-3.x
OUT_host= host2.x host1.x
OUT_module= mod1.x modalloc.x nestModule.x
OUT_epi= autosyncall1.x finalize.x
OUT_bug= $(BUG403) $(BUG451) $(BUG457) $(BUG458) $(BUG459) $(BUG460) $(BUG472) $(BUG006) $(ISSUE064) $(BUG085)
BUG403= bug403.x bug403OK.x   #403 Problem FuncDefBlock.Finalize()
	                      # To avoid this problem, the current implementation
	                      # does not allow to use both CAF and XMP together.
BUG451= bug451.x
BUG457= sakagami457.x
BUG458= sakagami458-join.x  sakagami458-main.x    ## Type8 NG
BUG459= sakagami459-join.x  sakagami459-main.x   ## Type8 NG
BUG460= sakagami460-join.x sakagami460-2.x #bug460-3.x (err in SX because of bug479)
BUG472= bug472.x bug472G1.x bug472G2.x bb472.x bb472G1.x bb472G2.x
BUG006= bug006.x bug006OK.x
BUG085= bug085cat.x bug085.x
ISSUE064= nested_common_name.x nested_common_name-1.x nested_common_name-1-0-0.x nested_common_name-1-0.x nested_common_name-2.x nested_common_name-3.x block_rename_coarray.x

SPECIALS= # bug460get.x bug460put.x

OUT_MFILE=mf_use2.x

#VER7=iproc_ss.x iproc_mss.x

#OUT_err=entry1ERR

#host3 host2-1 ... problem in gfortran?

OUT= $(OUT_bug) $(OUT_host) $(OUT_module) $(OUT_epi) $(OUT_darg) #$(OUT_MFILE)

include ../Makefile.common

mf_use2.x:: mf_mod2.xmod

mf_mod2.xmod mf_mod2.o: mf_mod2.f90
	xmpf90 -c $(OPT) $<

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

bug085.x: bug085m.f90 bug085.f90
	xmpf90 $(OPT) bug085m.f90 -c
	xmpf90 $(OPT) bug085m.o bug085.f90 -o $@

sakagami458-main.x: sakagami458-mod.f90 sakagami458-main.f90
	xmpf90 $(OPT) sakagami458-mod.f90 -c
	xmpf90 $(OPT) sakagami458-main.f90 -c
	xmpf90 $(OPT) sakagami458-mod.o sakagami458-main.o -o $@

sakagami459-main.x: sakagami459-mod.f90 sakagami459-main.f90
	xmpf90 $(OPT) sakagami459-mod.f90 -c
	xmpf90 $(OPT) sakagami459-main.f90 -c
	xmpf90 $(OPT) sakagami459-mod.o sakagami459-main.o -o $@

