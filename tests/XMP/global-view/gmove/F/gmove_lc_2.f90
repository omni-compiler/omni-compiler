program gmove_lc_2

  character*25 tname
  call gmove_lc_1a1t_bc(tname)
  call gmove_lc_1a1t_bc_off(tname)
  call gmove_lc_1a1t_b(tname)
  call gmove_lc_1a1t_b_h(tname)
  call gmove_lc_1a1t_b_off(tname)
  call gmove_lc_1a1t_c(tname)
  call gmove_lc_1a1t_c_h(tname)
  call gmove_lc_1a1t_c_off(tname)
  call gmove_lc_1a1t_gb(tname)

end program

subroutine gmove_lc_1a1t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
integer xmp_node_num
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do i=1,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_bc_off(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(*)
!$xmp template tx(n+4)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp align a(i) with tx(i+4)
!$xmp align b(i) with tx(i+4)

!$xmp loop (i) on tx(i+4)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i+4)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i) on tx(i+4)
do i=2,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_bc_off"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_b(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do i=1,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_b_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do i=1,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_b_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_b_off(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(*)
!$xmp template tx(n+3)
!$xmp distribute tx(block) onto p
!$xmp align a(i) with tx(i+3)
!$xmp align b(i) with tx(i+3)

!$xmp loop (i) on tx(i+3)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i+3)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i) on tx(i+3)
do i=2,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_b_off"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_c(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i)-a(i))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_c_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
integer a(n),b(n)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with tx(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(1:n)=a(1:n)

ierr=0
!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i)-a(i))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_c_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_c_off(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
!$xmp nodes p(*)
!$xmp template tx(n+3)
!$xmp distribute tx(cyclic) onto p
!$xmp align a(i) with tx(i+3)
!$xmp align b(i) with tx(i+3)

!$xmp loop (i) on tx(i+3)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on tx(i+3)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n)

ierr=0
!$xmp loop (i) on tx(i+3)
do i=2,n
  ierr=ierr+abs(b(i)-a(i))
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_c_off"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_1a1t_gb(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n),b(n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(gblock(mx)) onto p
!$xmp distribute ty(gblock(my)) onto p
!$xmp align a(i) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do i=1,n
  a(i)=i
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(:)=a(:)

ierr=0
!$xmp loop (i) on ty(i)
do i=1,n
  ierr=ierr+abs(b(i)-i)
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_1a1t_gb"
call chk_int(tname, ierr)

end subroutine

