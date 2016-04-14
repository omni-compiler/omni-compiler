  implicit real (a-h,o-z)
!!   include "xmp_coarray.h"
  real :: b(1:2,3:5)[6:*]
  integer buf1(1), buf3(3)
  real, allocatable :: a(:,:)[:,:,:]

  me=this_image()

  allocate (a(1:2,3:5)[6:7,8:10,-3:*])

  nerr=0
  if (6 /= lcobound(b,1)) nerr=nerr+1
  if (8 /= ucobound(b,1)) nerr=nerr+1   !! if num_images()==3

  buf1=lcobound(b)
  if (buf1(1) /= 6)  nerr=nerr+1
  buf1=ucobound(b)
  if (buf1(1) /= 8)  nerr=nerr+1   !! if num_images()==3

  if (6 /= lcobound(a,1)) nerr=nerr+1
  if (7 /= ucobound(a,1)) nerr=nerr+1
  if (8 /= lcobound(a,2)) nerr=nerr+1
  if (10 /= ucobound(a,2)) nerr=nerr+1
  if (-3 /= lcobound(a,3)) nerr=nerr+1
  if (-3 /= ucobound(a,3)) nerr=nerr+1   !! if num_images()<=6

  buf3=lcobound(a)
  if (buf3(1) /= 6) nerr=nerr+1
  if (buf3(2) /= 8) nerr=nerr+1
  if (buf3(3) /= -3) nerr=nerr+1
  buf3=ucobound(a)
  if (buf3(1) /= 7) nerr=nerr+1
  if (buf3(2) /= 10) nerr=nerr+1
  if (buf3(3) /= -3) nerr=nerr+1   !! if num_images()<=6

  if (nerr /= 0) then
     write(*,100) me,nerr
  else
     write(*,101) me
  endif

100 format("[",i0,"] NG nerr=",i0)
101 format("[",i0,"] OK")

  end

