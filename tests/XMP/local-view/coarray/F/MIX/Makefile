#OPT= --debug
#OPT= -g --debug --verbose
NPE= 8

#fatal: cannot convert Xcode to XcodeML.
#costmtInSpdo.x costmtInSpdo-2.x

#---- TASK block
OUT_TASK=this_image.x this_image_subnodes.x this_image3.x sync_task.x 
#reduct_subset.x ###mpirun noticed that process rank 2 with PID 15921 on node slave05 exited on signal 11 (Segmentation fault).
#this_image2.x: too complicated

#---- COARRAY directive
OUT_DIR=coarray_dir0.x coarray_dir1.x coarray_dir2.x coarray_dir_ti1.x coarray_dir_ti2.x ti_outofset.x block_coarray_dir0.x

#---- IMAGE directive
OUT_IMG=image_dir0.x image_dir1.x block_image_dir1.x #image_dir_sync1.x #not yet

#---- ALLOC/DEALLOC
OUT_ALO= autodealloc1.x alloc_subnodes2.x
#alloc_subnodes.x ###Operating system error: Cannot allocate memory
#coarraydir_allocatable3.x  #restriction coarray dir for allocatable coarray
#coarraydir_allocatable2-mod.x also freezed on ipv-conduit
#coarraydir_allocatable2.x freezed on ipv-couduit

#autodealloc0.x  restriction  cf. ../ALLOC/allo4.f90
#coarraydir_allocatable.x  buggy.  cf. bug4.f90

#---- dimensions
OUT_DIM=coindex2d.x nodes2d.x

#coarray_dir_inner.f90  #image_dir.x

OUT= $(OUT_DIR) $(OUT_TASK) $(OUT_IMG) $(OUT_ALO) $(OUT_DIM)


include ../Makefile.common
