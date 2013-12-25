program tgmove
integer i,n=9
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

print *, 'max error=',ierr

contains
end
