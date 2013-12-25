program tgmove
integer i,n=8
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp distribute ty(cyclic,cyclic,cyclic) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(i0,i1,i2) with ty(i0,i1,i2)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i0,i1,i2) on ty(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i0,i1,i2) on ty(i0,i1,i2)
do i2=2,5
  do i1=2,5
    do i0=2,5
      ierr=ierr+abs(b(i0,i1,i2)-(i0+3)-(i1+3)-(i2+3))
      print *, 'i0=',i1,'i1=',i1,'i2=',i2,'b=',b(i0,i1,i2)
    end do
  end do
end do

print *, 'max error=',ierr

contains
end
