type xmp_desc
  sequence
  integer*8 :: desc
end type xmp_desc

interface
integer(4) function xmp_get_mpi_comm()
  end function xmp_get_mpi_comm

integer(4) function xmp_num_nodes()
  end function xmp_num_nodes

integer(4) function xmp_node_num()
  end function xmp_node_num

integer(4) function xmp_all_num_nodes()
  end function xmp_all_num_nodes

integer(4) function xmp_all_node_num()
  end function xmp_all_node_num

real(8) function xmp_wtime()
  end function xmp_wtime

real(8) function xmp_wtick()
  end function xmp_wtick

end interface
