program tgmove
integer i
integer,parameter :: n=8
integer a(n),b(n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(gblock(mx)) onto p
!$xmp distribute ty(gblock(my)) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(:)=a(:)

ierr=0
!$xmp loop (i) on ty(i)
do i=1,n
  ierr=ierr+abs(b(i)-i)
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
