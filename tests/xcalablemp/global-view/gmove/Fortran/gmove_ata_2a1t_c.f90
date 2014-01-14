program tgmove
integer :: i,j
integer,parameter ::n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,i) with tx(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (j) on tx(j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (j) on tx(j)
do j=1,n
!$xmp loop (i) on tx(i)
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
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
