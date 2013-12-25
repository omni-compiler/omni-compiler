program tgmove
integer i,n=10
integer a(n),b(n)
!$xmp nodes p(3)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:5)=a(5:8)

ierr=0
!$xmp loop (i) on ty(i)
do i=2,5
  ierr=ierr+abs(b(i)-(i+3))
  print *, 'i=',i,'b(i)=',b(i)
end do

print *, 'max error=',ierr

contains
end
