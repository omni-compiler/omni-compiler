program tgmove
integer i,n=8
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
do i=1,n
  ierr=ierr+abs(b(i)-i)
end do

print *, 'max error=',ierr

contains
end
