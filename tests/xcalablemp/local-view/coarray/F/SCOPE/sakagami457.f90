module param
integer, parameter :: lx=100, ly=50
end module
include 'xmp_coarray.h'
use param
real*8 :: a(lx,ly)[*]
do i = 1, lx
   a(i,1) = i
end do
syncall
nerr=0
eps=0.000001
if (abs(a(1,1)-a(1,1)[2])>eps) nerr=nerr+1
if (abs(a(2,1)-a(2,1)[1])>eps) nerr=nerr+1
if (abs(a(3,1)-a(3,1)[3])>eps) nerr=nerr+1
if (abs(a(lx,1)-a(lx,1)[1])>eps) nerr=nerr+1

call final_msg(nerr,this_image())
stop
end

  subroutine final_msg(nerr,me)
    if (nerr==0) then 
       print '("[",i0,"] OK")', me
    else
       print '("[",i0,"] number of NGs: ",i0)', me, nerr
    end if
    return
  end subroutine final_msg
