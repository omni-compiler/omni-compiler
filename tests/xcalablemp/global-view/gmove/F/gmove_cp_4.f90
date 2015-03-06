program gmove_cp_4

  character*25 tname
  call gmove_cp_2a2t_b_c(tname)
  call gmove_cp_2a2t_bc(tname)
  call gmove_cp_2a2t_b(tname)
  call gmove_cp_2a2t_c(tname)

end program

subroutine gmove_cp_2a2t_b_c(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
!$xmp loop (i,j) on ty(i,j)
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3)-(j+3))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_2a2t_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_2a2t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with tx(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3+j+3))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_2a2t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_2a2t_b(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(block,block) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with tx(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3+j+3))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_2a2t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_2a2t_c(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with tx(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3+j+3))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_2a2t_c"
call chk_int(tname, ierr)

end subroutine

