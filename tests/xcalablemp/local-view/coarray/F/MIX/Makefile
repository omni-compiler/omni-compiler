#OPT= --debug
#OPT= -g --debug --verbose
NPE= 8

#fatal: cannot convert Xcode to XcodeML.
#costmtInSpdo.x costmtInSpdo-2.x

#---- TASK block
OUT_TASK=this_image.x this_image_subnodes.x this_image3.x sync_task.x  reduct_subset.x #this_image2.x: too complicated

#---- COARRAY directive
OUT_DIR=coarray_dir0.x coarray_dir1.x coarray_dir2.x coarray_dir_ti1.x coarray_dir_ti2.x ti_outofset.x

#---- IMAGE directive
OUT_IMG=image_dir0.x #image_dir_sync1.x #not yet

#coarray_dir_inner.f90  #image_dir.x

OUT= $(OUT_DIR) $(OUT_TASK) $(OUT_IMG)


include ../Makefile.common
