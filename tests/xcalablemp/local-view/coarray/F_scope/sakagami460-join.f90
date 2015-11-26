module alloc
include 'xmp_coarray.h'
parameter(lx=100,ly=50)
real*8, allocatable :: a(:,:)[:]
contains
subroutine allocdata
allocate ( a(lx,ly)[*] )
print *, "        ** 0","size=",size(a)
return
end
end module

use alloc
real :: b[*]
call allocdata

!!!!!!!!!!!!!!!!
print *, "        ** 1","size=",size(a)
!!!!!!!!!!!!!!!!
me=this_image()
do j=1,ly
   do i=1,lx
      a(i,j)=me*lx
   enddo
enddo
!!!!!!!!!!!!!!!!
print *, "        ** 1.5"
!!!!!!!!!!!!!!!!
b=sum(a)
!!!!!!!!!!!!!!!!
print *, "        ** 2"
!!!!!!!!!!!!!!!!
syncall

do i = 1, num_images()
  asum = asum + b[i]
end do
syncall

!!!!!!!!!!!!!!!!
print *, "        ** 3"
!!!!!!!!!!!!!!!!
nerr=0
eps=0.000001
ni = num_images()
nsum = 5050*50*ni*(ni+1)/2
if (abs(asum-nsum)>eps) then
   nerr=nerr+1
   print '("[",i0,"] sum should be about ",i0," but ",f10.3)', me, nsum, asum
endif

call final_msg(nerr, me)
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
