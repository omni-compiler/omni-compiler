subroutine hello_2(a, b) bind(C)
  implicit none
  include 'xmp_lib.h'
  integer(8), intent(in) :: a(3), b(3)
  
  write(*,*) xmp_node_num(), a(:), b(:)
end subroutine hello_2
