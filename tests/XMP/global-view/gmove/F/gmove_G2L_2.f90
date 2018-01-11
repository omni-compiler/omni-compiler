program gmove_G2L_2

  character*25 tname
  call gmove_G2L_1a1t_b(tname)
  call gmove_G2L_1a1t_bc(tname)

end program

subroutine gmove_G2L_1a1t_b(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
do i=1,n
  ierr=ierr+abs(b(i)-i)
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_G2L_1a1t_bc(tname)

character(*) tname
integer :: i
integer,parameter :: n=8
integer a(n),b(n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp align a(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:5)=a(5:8)

ierr=0
do i=2,5
  ierr=ierr+abs(b(i)-(i+3))
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a1t_bc"
call chk_int(tname, ierr)

end subroutine

