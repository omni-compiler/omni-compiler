module alloc
  include 'xmp_coarray.h'
  parameter(lx=100,ly=50)
  real*8, allocatable :: a(:,:)[:]
contains
  subroutine allocdata
    allocate ( a(lx,ly)[*] )
    return
  end subroutine allocdata
end module

use alloc
call allocdata

a(1,1)=lx


stop
end

