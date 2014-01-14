program tgmove
integer i,n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with tx(i,*)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

do j=1,n
!$xmp loop (i) on tx(i,*)
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:2,1:2)=a(1:2,1:2)

ierr=0
do j=1,2
!$xmp loop (i) on tx(i,*)
  do i=1,2
    ierr=ierr+abs(b(i,j)-i-j)
  end do
end do

!$xmp reduction (max:ierr)
irank=xmp_node_num()
if (irank==1) then
  print *, 'max error=',ierr
endif
!call exit(ierr)

stop
end
