program tgmove
integer i
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,2/)
integer xmp_node_num
!$xmp nodes p(4)
!$xmp nodes q(2,2)
!$xmp template tx(n)
!$xmp template ty(n,n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(gblock(m),gblock(m)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i,j) on ty(i,j)
do j=2,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp barrier
!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
