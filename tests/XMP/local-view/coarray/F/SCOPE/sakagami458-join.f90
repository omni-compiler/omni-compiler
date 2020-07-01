module sdata2
!! include 'xmp_coarray.h'
parameter(lx=100,ly=50)
real*8 :: a(100,50)[*]
end module

use sdata2
do i = 1, lx
   a(i,1)[2] = i
end do
syncall

nerr=0
eps=0.000001
if (abs(1-a(1,1)[2])>eps) nerr=nerr+1
if (abs(2-a(2,1)[2])>eps) nerr=nerr+1
if (abs(3-a(3,1)[2])>eps) nerr=nerr+1
if (abs(lx-a(lx,1)[2])>eps) nerr=nerr+1

call final_msg(nerr, this_image())
stop
end

  subroutine final_msg(nerr,me)
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
       call exit(1)
    end if
    return
  end subroutine final_msg
