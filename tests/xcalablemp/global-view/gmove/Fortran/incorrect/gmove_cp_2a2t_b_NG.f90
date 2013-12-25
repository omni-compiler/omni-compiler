program tgmove
integer i,n=8
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
b(4:7,4:7)=a(1:4,1:4)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=4,7
  do i=4,7
    ierr=ierr+abs(b(i,j)-(i-3+j-3))
  end do
end do

print *, 'max error=',ierr

contains
end
