program tgmove
integer i,n=8
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
!$xmp loop (i,j) on ty(i,j)
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3)-(j+3))
    print *, 'i=',i,'j=',j,'b(i,j)=',b(i,j)
  end do
end do

print *, 'max error=',ierr

contains
end
