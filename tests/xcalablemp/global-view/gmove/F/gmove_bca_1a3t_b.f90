program tgmove
integer :: i
integer,parameter :: n=4
integer a(n),b(n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i1) with tx(*,i1,*)
!$xmp align b(i2) with tx(*,*,i2)

!$xmp loop (i1) on tx(*,i1,*)
do i1=1,n
  a(i1)=i1
end do

!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  b(i2)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

myrank=xmp_node_num()
ierr=0
!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  ierr=ierr+abs(b(i2)-i2)
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
