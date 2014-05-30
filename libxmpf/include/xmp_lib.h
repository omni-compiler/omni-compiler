type xmp_desc
  sequence
  integer*8 :: desc
end type xmp_desc

interface
integer(4) function xmp_num_nodes()
  end function xmp_num_nodes
end interface
