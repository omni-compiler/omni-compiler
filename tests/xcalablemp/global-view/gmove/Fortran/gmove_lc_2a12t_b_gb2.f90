program tgmove
integer i
integer,parameter :: n=4
integer a(n,n),b(n)
integer m(2)=(/2,2/)
integer xmp_node_num
!$xmp nodes p(4)
!$xmp nodes q(4)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(gblock(m)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n,1)

ierr=0
!$xmp loop (i) on ty(i)
do i=2,n
  ierr=ierr+abs(b(i)-i-1)
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
