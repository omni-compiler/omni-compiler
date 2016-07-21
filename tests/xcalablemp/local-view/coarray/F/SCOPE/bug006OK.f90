!! test for allocation alignment
!! for comparison with bug006.f90

module z006OK
real*8, save :: a(100,50)[*]
end module

use z006OK

real :: b[*], dummy[*]

me=this_image()
do j=1,50
   do i=1,100
      a(i,j)=me*i
   enddo
enddo
b=sum(a)

if (b==252500*me) then
   write(*,*) "OK"
else
   write(*,*) "NG"
endif

end
