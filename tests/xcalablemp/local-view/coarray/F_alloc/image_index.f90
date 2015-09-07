  include "xmp_coarray.h"
  real :: ss(1:2,3:5)[6:*]
  integer sub(1)
  real, allocatable :: a(:,:)[:,:,:]

  me=this_image()

  sub(1)=8
  nnnn = image_index(ss, sub)
  write(*,*) nnnn, 3

  call foo(a)
!  write(*,*) image_index(a, [8,9,3]), 2+4*1+4*3*6

  contains
    subroutine foo(a)
      real, allocatable :: a(:,:)[:,:,:]
      write(*,*) "hi!"
!!      allocate (a(1:2,3:5)[6:9,8:10,-3:*])
      return
    end subroutine foo
  end
