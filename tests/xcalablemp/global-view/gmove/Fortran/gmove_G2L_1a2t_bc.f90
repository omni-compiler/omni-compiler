program tgmove
integer,parameter :: n=4
integer a(n,n), b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,i1) with tx(*,i1)

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
do i1=2,n
  do i0=2,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (max:ierr)
irank=xmp_node_num()
if (irank==1) then
  print *, 'max error=',ierr
endif
!call exit(ierr)

stop
end program tgmove
