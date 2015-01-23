program tgmove
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i0,i1) with tx(i0,i1,*)
!$xmp align b(i1,i2) with tx(*,i1,i2)

!$xmp loop (i0,i1) on tx(i0,i1,*)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i1,i2) on tx(*,i1,i2)
do i2=1,n
  do i1=1,n
    b(i1,i2)=0
  end do
end do

!$xmp gmove
b(:,:)=a(:,:)

myrank=xmp_node_num()
ierr=0
!$xmp loop (i1,i2) on tx(*,i1,i2)
do i2=1,n
  do i1=1,n
    ierr=ierr+abs(b(i1,i2)-i1-i2)
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
