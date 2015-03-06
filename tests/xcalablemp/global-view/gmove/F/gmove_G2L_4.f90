program gmove_G2L_4

  character*25 tname
  call gmove_G2L_1a2t_bc(tname)
  call gmove_G2L_2a2t_bc(tname)

end program

subroutine gmove_G2L_1a2t_bc(tname)

character(*) tname
integer,parameter :: n=4
integer a(n,n), b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic(2),cyclic(2)) onto p
!$xmp align a(*,i1) with tx(*,i1)

!$xmp loop (i1) on tx(*,i1)
do i1=1,n
  do i0=1,n
    a(i0,i1)=i0+i1
  end do
end do

do i1=1,n
  do i0=1,n
    b(i0,i1)=0
  end do
end do

!$xmp gmove
b(2:n,2:n)=a(2:n,2:n)

ierr=0
do i1=2,n
  do i0=2,n
    ierr=ierr+abs(b(i0,i1)-i0-i1)
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a2t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_G2L_2a2t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(i,j)

!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:5,2:5)=a(5:8,5:8)

ierr=0
do j=2,5
  do i=2,5
    ierr=ierr+abs(b(i,j)-(i+3+j+3))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_2a2t_bc"
call chk_int(tname, ierr)

end subroutine

