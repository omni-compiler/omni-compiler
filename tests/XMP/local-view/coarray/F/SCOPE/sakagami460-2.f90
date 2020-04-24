module alloc2
!! include 'xmp_coarray.h'
parameter(lx=100,ly=50)
!!real*8, allocatable :: a(:,:)[:]
real*8, save :: a(lx,ly)[*]
contains
subroutine allocdata
!!allocate ( a(lx,ly)[*] )
return
end
end module

use alloc2
real :: b[*]
call allocdata

me=this_image()
do j=1,ly
   do i=1,lx
      a(i,j)=me*i
   enddo
enddo
b=sum(a)
syncall

asum=0.0
do i = 1, num_images()
  asum = asum + b[i]
end do
syncall

nerr=0
eps=0.000001
ni = num_images()
nsum = 5050*50*ni*(ni+1)/2
if (abs(asum-nsum)>eps) then
   nerr=nerr+1
   print '("[",i0,"] sum should be about ",i0," but ",f20.2)', me, nsum, asum
endif

call final_msg(nerr, me)
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
