#OPT= -g --debug
#OPT= -g --debug --verbose

NPE=3   # For image_index0.x and this_iamge.x, NPE>=8 is suitable.
$NPE=8
#NPE=18

OUT_decl= decl3.x decl1.x
OUT_size= size1.x size1-2.x #sample #size1-err
#OUT_rt  =  inquiry
OUT_NG =  allo2NG.x

OUT_allo= allo1.x allo3.x 	
#allo4.x restriction: excessive auto-dealloc of allocatable-dummy-coarray

OUT_epi= allodeallo.x #epi2 #epi1
OUT_etc=allo_pipo3.x allo_loop2.x #allo_hiloop.x #allo_hicall #char12  host_assoc
OUT_debug=bug418.x bug418-2.x bug468.x bug015.x 020.x

#OUT_err= init1ERR.x recursiveERR.x

OUT_interface=interface.x  # assumed-shape coarray
OUT_stat=#allo_stat.x

OUT = $(OUT_epi) $(OUT_etc) $(OUT_decl) $(OUT_size) $(OUT_allo) $(OUT_debug)

include ../Makefile.common

