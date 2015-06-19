#OPT= -g --debug
#OPT= -g --debug --verbose
NPE=3

OUT_decl= decl3.x #decl1
#OUT_size= size1 size1-2 #size2 #sample #size1-err
### size2: assumed shape dummy coarray
#OUT_rt  =  inquiry
OUT_NG =  allo2NG.x

OUT_allo= allo1.x allo3.x
OUT_epi= allodeallo.x #epi2 #epi1
OUT_etc=allo_pipo3.x allo_loop2.x allo_hiloop.x #allo_hicall #char12  host_assoc

#OUT_err= init1ERR recursiveERR

OUT_stat=#allo_stat

OUT = $(OUT_epi) $(OUT_etc) $(OUT_decl) $(OUT_size) $(OUT_allo) $(OUT_rt)

include ../F/Makefile.common

