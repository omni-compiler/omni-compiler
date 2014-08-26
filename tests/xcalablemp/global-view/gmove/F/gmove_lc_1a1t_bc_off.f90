program tgmove
include 'xmp_lib.h'
integer i
integer,parameter :: n=8
integer a(n),b(n)
integer irank
!$xmp nodes p(*)
!$xmp template tx(n+4)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp align a(i) with tx(i+4)
!$xmp align b(i) with tx(i+4)

irank=xmp_node_num()

!$xmp loop (i) on tx(i+4)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i+4)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i) on tx(i+4)
do i=2,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

end program tgmove
