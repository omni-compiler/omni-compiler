program tgmove
integer i
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,2/)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!!$xmp distribute tx(block,block) onto p
!$xmp distribute tx(gblock(m),gblock(m)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with tx(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=2,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp barrier
!$xmp reduction (max:ierr)
irank=xmp_node_num()
if (irank==1) then
  print *, 'max error=',ierr
endif
!call exit(ierr)

stop
end program tgmove
