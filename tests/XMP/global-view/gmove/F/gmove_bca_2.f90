program gmove_

  character*25 tname
  call gmove_bca_1a2t_b3(tname)
  call gmove_bca_1a2t_b4(tname)
  call gmove_bca_1a2t_b5(tname)
  call gmove_bca_1a2t_b6(tname)

end program

subroutine gmove_bca_1a2t_b3(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2,1)
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
tname="gmove_bca_1a2t_b3"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_1a2t_b4(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(1,2)
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
tname="gmove_bca_1a2t_b4"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_1a2t_b5(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(1,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i1) with tx(*,i1)
!$xmp align b(i0) with tx(i0,*)

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  a(i1)=i1
end do

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  b(i0)=0
end do

!$xmp gmove
b(:)=a(:)

ierr=0
!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  ierr=ierr+abs(b(i0)-i0)
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_1a2t_b5"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_1a2t_b6(tname)

character(*) tname
integer :: i
integer,parameter :: n=4
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(block,*) onto p
!$xmp distribute ty(*,block) onto p
!$xmp align a(i0) with tx(i0,*)
!$xmp align b(i1) with ty(*,i1)

!$xmp loop (i0) on tx(i0,*)
do i0=1,n
  a(i0)=i0
end do

!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  b(i1)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

myrank=xmp_node_num()
ierr=0
!$xmp loop (i1) on ty(*,i1)
do i1=1,n
  ierr=ierr+abs(b(i1)-i1)
end do

!$xmp reduction (+:ierr)
tname="gmove_bca_1a2t_b6"
call chk_int(tname, ierr)

end subroutine

