program gmove_bca_16

  character*25 tname
  call gmove_bca_4a4t_bc(tname)
  call gmove_bca_4a4t_b2(tname)

end program

subroutine gmove_bca_4a4t_bc(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n,n),b(n,n,n,n)
!$xmp nodes p(2,2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2,i3) with tx(i0,i1,i2,i3)
!$xmp align b(*,i1,i2,i3) with tx(*,i1,i2,i3)

irank=xmp_node_num()

!$xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        a(i0,i1,i2,i3)=i0+i1+i2+i3
      end do
    end do
  end do
end do

!$xmp loop (i1,i2,i3) on tx(*,i1,i2,i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        b(i0,i1,i2,i3)=0
      end do
    end do
  end do
end do

!$xmp gmove
b(2:n,2:n,2:n,2:n)=a(2:n,2:n,2:n,2:n)

ierr=0
!$xmp loop (i1,i2,i3) on tx(*,i1,i2,i3)
do i3=2,n
  do i2=2,n
    do i1=2,n
      do i0=2,n
        ierr=ierr+abs(b(i0,i1,i2,i3)-i0-i1-i2-i3)
      end do
    end do
  end do
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_4a4t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_4a4t_b2(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n,n),b(n,n,n,n)
!$xmp nodes p(2,2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp distribute tx(block,block,block,block) onto p
!$xmp align a(i0,i1,i2,i3) with tx(i0,i1,i2,i3)
!$xmp align b(*,i1,i2,*) with tx(*,i1,i2,*)

!$xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        a(i0,i1,i2,i3)=i0+i1+i2+i3
      end do
    end do
  end do
end do

do i3=1,n
!$xmp loop (i1,i2) on tx(*,i1,i2,*)
  do i2=1,n
    do i1=1,n
      do i0=1,n
        b(i0,i1,i2,i3)=0
      end do
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n,1:n)=a(1:n,1:n,1:n,1:n)

ierr=0
do i3=1,n
!$xmp loop (i1,i2) on tx(*,i1,i2,*)
  do i2=1,n
    do i1=1,n
      do i0=1,n
        ierr=ierr+abs(b(i0,i1,i2,i3)-i0-i1-i2-i3)
      end do
    end do
  end do
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_4a4t_b2"
call chk_int(tname, ierr)

end subroutine

