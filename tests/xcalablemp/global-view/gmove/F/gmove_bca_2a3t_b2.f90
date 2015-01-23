program tgmove
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(block,*,block) onto p
!$xmp distribute ty(*,block,block) onto p
!$xmp align a(i0,*) with tx(i0,*,*)
!$xmp align b(*,i2) with ty(*,*,i2)

do i1=1,n
!$xmp loop (i0) on tx(i0,*,*)
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i2) on ty(*,*,i2)
do i2=1,n
  do i1=1,n
    b(i1,i2)=0
  end do
end do

!$xmp gmove
b(:,:)=a(:,:)

myrank=xmp_node_num()
ierr=0
!$xmp loop (i2) on ty(*,*,i2)
do i2=1,n
  do i1=1,n
    ierr=ierr+abs(b(i1,i2)-i1-i2)
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
