parameter(lx=100,ly=lx/2)
! parameter(lx=100,ly=50)
include 'xmp_coarray.h'
real*8 :: a(lx,ly)[*]
do i = 1, lx
   a(i,1) = i
end do
stop
end
