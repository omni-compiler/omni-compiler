  module mmm
!!     include 'xmp_coarray.h'
    integer aaa[*]
  end module mmm

  module mm1
    integer bbb
  end module mm1

  module mm2
    use mmm
    integer bbb[*]
  end module mm2

