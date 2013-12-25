program tgmove
integer i,n=8
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
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

contains
end
