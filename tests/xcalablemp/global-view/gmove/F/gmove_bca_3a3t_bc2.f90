program tgmove
integer :: i,irank,xmp_node_num
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i1,i2) with tx(*,i1,i2)

irank=xmp_node_num()

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i1,i2) on tx(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(:,:,:)=a(:,:,:)

ierr=0
!$xmp loop (i1,i2) on tx(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i1,i2)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)
stop
end
