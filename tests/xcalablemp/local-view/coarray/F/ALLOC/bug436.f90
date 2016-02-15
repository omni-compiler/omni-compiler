program zz
  include "xmp_coarray.h"
  real, allocatable :: a(:,:)[:,:,:]

  allocate (a(1:2,3:5)[6:9,8:10,-3:*])
  write(*,*) "ho!"
  call foo(a)
  write(*,*) "hu!"
end program

subroutine foo(a)
  include "xmp_coarray.h"
  real, allocatable :: a(:,:)[:,:,:]
  write(*,*) "hi!"
  return
end subroutine foo

