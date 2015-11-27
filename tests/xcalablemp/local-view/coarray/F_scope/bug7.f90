module alloc
include 'xmp_coarray.h'
parameter(lx=100,ly=50)
real*8, allocatable :: a(:,:)[:]
contains
subroutine allocdata
allocate ( a(lx,ly)[*] )
return
end
end module

use alloc
real :: b[*]
call allocdata

me=this_image()
do j=1,ly
   do i=1,lx
      a(i,j)=me*lx
   enddo
enddo
b=sum(a)

if (abs(sum-nsum)>eps) continue

stop
end

