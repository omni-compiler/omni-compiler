program tgmove
integer :: i,irank,xmp_node_num
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp template ty(n,n,n,n)
!$xmp distribute tx(block,block,*,block) onto p
!$xmp distribute ty(block,*,block,block) onto p
!$xmp align a(*,i1,i3) with tx(*,i1,*,i3)
!$xmp align b(i0,i2,*) with ty(i0,*,i2,*)

irank=xmp_node_num()

!$xmp loop (i1,i3) on tx(*,i1,*,i3)
do i3=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i3)=i0+i1+i3
    end do
  end do
end do

do i3=1,n
!$xmp loop (i0,i2) on ty(i0,*,i2,*)
  do i2=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
do i3=1,n
!$xmp loop (i0,i2) on ty(i0,*,i2,*)
  do i2=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i3)-i0-i2-i3)
    end do
  end do
end do

!$xmp reduction (+:ierr)
call chk_int(ierr)

stop
end program
