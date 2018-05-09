!!   include "xmp_coarray.h"
  real :: ss(1:2,3:5)[6:*]
  integer sub(1)
  real, allocatable :: a(:,:)[:,:,:]

  me=this_image()
  nerr=0

  sub(1)=8
  nnnn = image_index(ss, sub)
  if (nnnn.ne.3) nerr=nerr+1

  call foo(a)
  write(*,*) image_index(a, [8,10,-2]), 3+4*2+4*3*1

  contains
    subroutine foo(a)
      real, allocatable :: a(:,:)[:,:,:]
      write(*,*) "hi!"
      allocate (a(1:2,3:5)[6:9,8:10,-3:*])
      return
    end subroutine foo
  end

!!! problem about host-association
