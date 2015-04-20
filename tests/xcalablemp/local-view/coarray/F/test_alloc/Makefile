#OPT= -g --debug
#OPT= -g --debug --verbose
NPE=3

OUT_decl= decl1
#OUT_size= size1 size1-2 #size2 #sample #size1-err
### size2: assumed shape dummy coarray
#OUT_rt  =  inquiry
OUT_NG =  allo2NG

OUT_allo= allo1 allo3 
OUT_epi= allodeallo #epi2 #epi1
OUT_etc=allo_pipo3 allo_loop2 allo_hiloop #allo_hicall #char12  host_assoc

#OUT_err= init1ERR recursiveERR

OUT_stat=#allo_stat

OUT = $(OUT_epi) $(OUT_etc) $(OUT_decl) $(OUT_size) $(OUT_allo) $(OUT_rt)

include ../Makefile.common

