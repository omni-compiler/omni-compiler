program tgmove
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=2,5
  do j=2,5
    do i=2,5
      ierr=ierr+abs(b(i,j,k)-(i+3+j+3+k+3))
    end do
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
