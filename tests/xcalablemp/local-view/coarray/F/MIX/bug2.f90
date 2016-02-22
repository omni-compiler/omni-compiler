program test
  include 'xmp_lib.h'
!$xmp nodes pp(6)

  call test_sub(xmp_desc_of(pp))

contains
  subroutine test_sub(xx)
    type(xmp_desc):: xx
  end subroutine test_sub

end program test


