module param
integer, parameter :: lx=100, ly=50
end module
include 'xmp_coarray.h'
use param
real*8 :: a(lx,ly)[*]
do i = 1, lx
   a(i,1) = i
end do
stop
end
