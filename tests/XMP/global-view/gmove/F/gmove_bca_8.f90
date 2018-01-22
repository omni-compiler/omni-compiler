program gmove_bca_8

  character*25 tname
  call gmove_bca_1a3t_b(tname)
  call gmove_bca_2a2t_b_subcomm(tname)
  call gmove_bca_2a3t_b(tname)
  call gmove_bca_3a3t_b2(tname)
  call gmove_bca_3a3t_b(tname)
  call gmove_bca_3a3t_bc2(tname)
  call gmove_bca_3a3t_bc(tname)
  call gmove_bca_3a4t_b(tname)

end program

subroutine gmove_bca_1a3t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n),b(n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i1) with tx(*,i1,*)
!$xmp align b(i2) with tx(*,*,i2)

!$xmp loop (i1) on tx(*,i1,*)
do i1=1,n
  a(i1)=i1
end do

!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  b(i2)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

myrank=xmp_node_num()
ierr=0
!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  ierr=ierr+abs(b(i2)-i2)
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_1a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_2a2t_b_subcomm(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n,n),b(n,n)
!$xmp nodes p(8)
!$xmp nodes p1(2,2)=p(1:4)
!$xmp nodes p2(2,2)=p(5:8)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(block,block) onto p1
!$xmp distribute ty(block,block) onto p2
!$xmp align a(i0,i1) with tx(i0,i1)
!$xmp align b(*,i1) with ty(*,i1)

!$xmp loop (i0,i1) on tx(i0,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  do i0=1,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (+:ierr) on p2
tname="gmove_bca_2a2t_b_subcomm"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_2a3t_b(tname)

character(*) tname
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
tname="gmove_bca_2a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_b2(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i1,i2) with tx(*,i1,i2)

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
tname="gmove_bca_3a3t_b2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i1,i2) with tx(*,i1,i2)

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
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

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
tname="gmove_bca_3a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_bc2(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i1,i2) with tx(*,i1,i2)

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
tname="gmove_bca_3a3t_bc2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_bc(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i1,i2) with tx(*,i1,i2)

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
b(2:n,2:n,2:n)=a(2:n,2:n,2:n)

ierr=0
!$xmp loop (i1,i2) on tx(*,i1,i2)
do i2=2,n
  do i1=2,n
    do i0=2,n
      ierr=ierr+abs(b(i0,i1,i2)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_3a3t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a4t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp template ty(n,n,n,n)
!$xmp distribute tx(block,block,*,block) onto p
!$xmp distribute ty(block,*,block,block) onto p
!$xmp align a(*,i1,i3) with tx(*,i1,*,i3)
!$xmp align b(i0,i2,*) with ty(i0,*,i2,*)

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
tname="gmove_bca_3a4t_b"
call chk_int(tname, ierr)

end subroutine
