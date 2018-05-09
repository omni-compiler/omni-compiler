program gmove_bca_4

  character*25 tname
  call gmove_bca_1a2t_b(tname)
  call gmove_bca_1a2t_b2(tname)
  call gmove_bca_2a2t_b(tname)
  call gmove_bca_2a3t_b2(tname)

end program

subroutine gmove_bca_1a2t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i0) with tx(i0,*)
!$xmp align b(i1) with tx(*,i1)

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  a(i0)=i0
end do

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  b(i1)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i1) on tx(*,i1)
do i1=2,n
  ierr=ierr+abs(b(i1)-i1)
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_1a2t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_1a2t_b2(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i0) with tx(i0,*)
!$xmp align b(i0) with tx(i0,*)

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  a(i0)=i0
end do

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  b(i0)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i0) on tx(i0,*)
do i0=2,n
  ierr=ierr+abs(b(i0)-i0)
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_1a2t_b2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_2a2t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i0,i1) with tx(i0,i1)
!$xmp align b(*,i1) with tx(*,i1)

!$xmp loop (i0,i1) on tx(i0,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i1) on tx(*,i1)
do i1=2,n
  do i0=2,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_2a2t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_2a3t_b2(tname)

character(*) tname
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

ierr=0
!$xmp loop (i2) on ty(*,*,i2)
do i2=1,n
  do i1=1,n
    ierr=ierr+abs(b(i1,i2)-i1-i2)
  end do
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_2a3t_b2"
call chk_int(tname, ierr)

end subroutine
