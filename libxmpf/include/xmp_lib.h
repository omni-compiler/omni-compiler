type xmp_desc
  sequence
  integer*8 :: desc
end type xmp_desc

integer(4) xmp_get_mpi_comm
integer(4) xmp_num_nodes
integer(4) xmp_node_num
integer(4) xmp_all_num_nodes
integer(4) xmp_all_node_num
real(8) xmp_wtime
real(8) xmp_wtick
integer(4) xmp_array_ndims
integer(4) xmp_array_lbound
integer(4) xmp_array_ubound
integer(4) xmp_array_lsize
integer(4) xmp_array_ushadow
integer(4) xmp_array_lshadow
integer(4) xmp_array_lead_dim
integer(4) xmp_array_gtol
integer(4) xmp_align_axis
integer(4) xmp_align_offset
integer(4) xmp_align_replicated
integer(4) xmp_align_template
integer(4) xmp_template_fixed
integer(4) xmp_template_ndims
integer(4) xmp_template_lbound
integer(4) xmp_template_ubound
integer(4) xmp_dist_format
integer(4) xmp_dist_blocksize
integer(4) xmp_dist_gblockmap
integer(4) xmp_dist_nodes
integer(4) xmp_dist_axis
integer(4) xmp_nodes_ndims
integer(4) xmp_nodes_index
integer(4) xmp_nodes_size
integer(4) xmp_nodes_equiv
