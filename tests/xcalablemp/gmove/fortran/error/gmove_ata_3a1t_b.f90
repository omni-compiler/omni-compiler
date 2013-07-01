program tgmove
integer i,j,k,n=8
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i,*,*) with tx(i)
!$xmp align b(*,*,i) with tx(i)

do k=1,n
  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (k) on tx(k)
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
!$xmp loop (k) on tx(k)
do k=1,n
  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

print *, 'max error=',ierr

contains
end
