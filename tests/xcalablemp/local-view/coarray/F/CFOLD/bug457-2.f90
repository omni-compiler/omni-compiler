module param0
integer, parameter :: nn=333
end module

module param
use param0
integer, parameter :: lx=nn/3
end module

use param
!! include "xmp_coarray.h"
real*8 :: a(3:lx,nn)[*]

!!write(*,*) ubound(a,1)," should be 111."
!!write(*,*) size(a,1)," should be 109."
!!write(*,*) size(a,2)," should be 333."

nerr=0
if (ubound(a,1) /= 111) nerr=nerr+1
if (size(a,1) /= 109) nerr=nerr+1
if (size(a,2) /= 333) nerr=nerr+1

if (nerr==0) then 
   print '("[",i0,"] OK")', this_image()
else
   print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
end if

end
