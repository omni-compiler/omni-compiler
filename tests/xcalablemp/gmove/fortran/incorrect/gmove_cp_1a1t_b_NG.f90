program tgmove
integer i,n=8
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(4:7)=a(1:4)

ierr=0
!$xmp loop (i) on tx(i)
do i=4,7
  print *,'i=',i,'b=',b(i)
  ierr=ierr+abs(b(i)-(i-3))
end do

print *, 'max error=',ierr

contains
end
