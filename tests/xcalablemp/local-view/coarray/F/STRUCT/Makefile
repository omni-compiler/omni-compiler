#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_debug = 
OUT_basic = 
OUT_lev1  = 

# wait for issue#37
ifdef (issue037)
OUT_basic += struct_align.x put_compo1.x 
OUT_lev1  += lev1.x lev1-2.x
endif

# restciction on FJ compiler and Ver.3
ifneq ($(ONESIDED_LAYER),FJRDMA)
OUT_debug += issue033.x
OUT_basic += struct_align2.x alloc0.x
ifdef (issue037)
OUT_debug += issue033-2.x
OUT_basic += get_compo1.x
endif
endif

#alloc1NG.x get_compo2.x #get1.x #put1.x  putget1.x struct_align2NG.x
#OUT_arg    = aarg1.x
#OUT_mod    = mod1.x

OUT = $(OUT_debug) $(OUT_basic) $(OUT_lev1)
# $(OUT_arg) $(OUT_mod)

include ../Makefile.common
