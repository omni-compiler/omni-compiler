program tgmove
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i0) with tx(i0,*)
!$xmp align b(i0) with tx(i0,*)

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  a(i0)=i0
end do

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  b(i0)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i0) on tx(i0,*)
do i0=2,n
  ierr=ierr+abs(b(i0)-i0)
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
