program tgmove
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(gblock(mx),gblock(mx),gblock(mx)) onto p
!$xmp distribute ty(gblock(my),gblock(my),gblock(my)) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with ty(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on ty(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on ty(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-i-j-k)
    end do
  end do
end do

!$xmp reduction (max:ierr)
call chk_int(ierr)

stop
end program tgmove
