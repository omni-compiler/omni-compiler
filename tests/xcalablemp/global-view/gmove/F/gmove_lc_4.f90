program gmove_

  character*25 tname
  call gmove_lc_21a1t_b(tname)
  call gmove_lc_21a1t_b_gb2(tname)
  call gmove_lc_21a1t_b_gb(tname)
  call gmove_lc_21a2t_b(tname)
  call gmove_lc_2a12t_b_gb(tname)
  call gmove_lc_2a2t_bc(tname)
  call gmove_lc_2a2t_b(tname)
  call gmove_lc_2a2t_b_h(tname)
  call gmove_lc_2a2t_c(tname)
  call gmove_lc_2a2t_c_h(tname)
  call gmove_lc_2a2t_gb2(tname)
  call gmove_lc_2a2t_gb3(tname)
  call gmove_lc_2a2t_gb(tname)

end program

subroutine gmove_lc_21a1t_b(tname)

character(*) tname
integer i
integer,parameter :: n=4
integer a(n,n),b(n)
integer m(2)=(/2,2/)
!$xmp nodes p(4)
!$xmp nodes q(4)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(1,2:n)

ierr=0
!$xmp loop (i) on ty(i)
do i=2,n
  ierr=ierr+abs(b(i)-i-1)
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_21a1t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_21a1t_b_gb2(tname)

character(*) tname
integer i
integer,parameter :: n=4
integer a(n,n),b(n)
integer m(2)=(/2,2/)
!$xmp nodes p(4)
!$xmp nodes q(4)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(gblock(m)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(1,2:n)

ierr=0
!$xmp loop (i) on ty(i)
do i=2,n
  ierr=ierr+abs(b(i)-i-1)
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_21a1t_b_gb2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_21a1t_b_gb(tname)

character(*) tname
integer i
integer,parameter :: n=4
integer a(n,n),b(n)
integer m(2)=(/2,2/)
!$xmp nodes p(4)
!$xmp nodes q(4)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(gblock(m)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i) with ty(i)

!$xmp loop (i) on tx(i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i) on ty(i)
do i=1,n
  b(i)=0
end do

!$xmp gmove
b(2:n)=a(2:n,1)

ierr=0
!$xmp loop (i) on ty(i)
do i=2,n
  ierr=ierr+abs(b(i)-i-1)
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_21a1t_b_gb"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_21a2t_b(tname)

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
b(1:n,1)=a(1,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,1
  do i=1,n
    ierr=ierr+abs(b(i,j)-i-1)
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_21a2t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a12t_b_gb(tname)

character(*) tname
integer i
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,2/)
integer xmp_node_num
!$xmp nodes p(4)
!$xmp nodes q(2,2)
!$xmp template tx(n)
!$xmp template ty(n,n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(gblock(m),gblock(m)) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i) on tx(i)
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
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i,j) on ty(i,j)
do j=2,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a12t_b_gb"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_bc(tname)

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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_b(tname)

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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_b_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_b_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_c(tname)

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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_c_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_c_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_gb2(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(gblock(mx),gblock(mx)) onto p
!$xmp distribute ty(gblock(my),gblock(my)) onto p
!$xmp align a(i,j) with tx(j,i)
!$xmp align b(i,j) with tx(j,i)

!$xmp loop (i,j) on tx(j,i)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_gb2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_gb3(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n),b(n,n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp template ty(n,n)
!$xmp distribute tx(gblock(mx),gblock(mx)) onto p
!$xmp distribute ty(gblock(my),gblock(my)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with tx(i,j)

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
b(1:n,1:n)=a(1:n,1:n)

ierr=0
!$xmp loop (i,j) on ty(i,j)
do j=1,n
  do i=1,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_gb3"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_2a2t_gb(tname)

character(*) tname
integer i
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,2/)
integer xmp_node_num
!$xmp nodes p(2,2)
!$xmp template tx(n,n)
!$xmp distribute tx(gblock(m),gblock(m)) onto p
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
b(2:n,2:n)=a(2:n,2:n)

ierr=0
!$xmp loop (i,j) on tx(i,j)
do j=2,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-a(i,j))
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_2a2t_gb"
call chk_int(tname, ierr)

end subroutine

