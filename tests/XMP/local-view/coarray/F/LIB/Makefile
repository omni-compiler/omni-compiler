#OPT= --debug
#OPT= -g --debug --verbose
NPE=3

OUT_inq=this_image.x this_image2.x this_image3.x cobound.x image_index0.x image_index1.x #image_index.x
OUT_bcast = bcast0.x bcast1.x bcast2.x bcast3.x bcast_arg2i8.x
OUT_reduct = reduct.x reductArg12.x reductArg123.x reductArg13.x reductArg1.x
#OUT_lock = lock.x  not supported yet: 
#                     type(lock_type) must be defined in iso_fortran_env...
#OUT_critical = critical.x  not supported yet

#OUT_atom=adef.x adef2.x 

OUT=$(OUT_atom) $(OUT_inq) $(OUT_bcast) $(OUT_reduct)


include ../Makefile.common

#reduct7.result: reduct7.x
#	export XMP_ONESIDED_HEAP_SIZE=70M; mpirun -np 7 $< 2>&1 | tee $@

#reduct7.result: reduct7.f90
#	xmpf90 $(OPT) -o $*.x $<
#	export XMP_ONESIDED_HEAP_SIZE=70M; mpirun -np 7 $*.x 2>&1 | tee $@

