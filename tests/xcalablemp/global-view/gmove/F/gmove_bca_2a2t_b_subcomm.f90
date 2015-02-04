program tgmove
integer :: i,irank,xmp_node_num
integer,parameter :: n=8
integer a(n,n),b(n,n)
!$xmp nodes p(8)
!$xmp nodes p1(2,2)=p(1:4)
!$xmp nodes p2(2,2)=p(5:8)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(block,block) onto p1
!$xmp distribute ty(block,block) onto p2
!$xmp align a(i0,i1) with tx(i0,i1)
!$xmp align b(*,i1) with ty(*,i1)

irank=xmp_node_num()

!$xmp loop (i0,i1) on tx(i0,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  do i0=1,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (+:ierr) on p2
call chk_int(ierr)

stop
end program
