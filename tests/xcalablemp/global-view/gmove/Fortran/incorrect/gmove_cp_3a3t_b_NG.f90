program tgmove
integer i,n=8
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
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
b(4:7,4:7,4:7)=a(1:4,1:4,1:4)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=4,7
  do j=4,7
    do i=4,7
      ierr=ierr+abs(b(i,j,k)-(i-3+j-3+k-3))
    end do
  end do
end do

print *, 'max error=',ierr

contains
end
