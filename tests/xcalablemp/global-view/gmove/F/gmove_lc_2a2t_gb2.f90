program tgmove
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(gblock(mx),gblock(mx)) onto p
!$xmp distribute ty(gblock(my),gblock(my)) onto p
!$xmp align a(i,j) with tx(j,i)
!$xmp align b(i,j) with tx(j,i)

!$xmp loop (i,j) on tx(j,i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp barrier
!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
