program tgmove
integer i
integer,parameter :: n=9
integer a(n),b(n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do i=1,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
