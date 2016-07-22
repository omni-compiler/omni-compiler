!! test for allocation alignment
!! Coarray a should be aligned with 8-byte boundary. see issues #6

module z006
real*8, save :: a(100,50)[*]
end module

use z006

real :: b[*]

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
