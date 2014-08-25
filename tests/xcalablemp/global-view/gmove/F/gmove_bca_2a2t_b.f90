program tgmove
integer :: i,irank,xmp_node_num
integer,parameter :: n=8
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i0,i1) with tx(i0,i1)
!$xmp align b(*,i1) with tx(*,i1)

irank=xmp_node_num()

!$xmp loop (i0,i1) on tx(i0,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i1) on tx(*,i1)
do i1=2,n
  do i0=2,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
