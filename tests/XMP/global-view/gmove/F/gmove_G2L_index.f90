program test_off

  character*25 tname
!$xmp nodes p(2,2)

call sub1_b_bc(tname)
call sub2_b_bc(tname)
call sub3_b_bc(tname)
call sub4_b_bc(tname)
call sub5_b_bc(tname)
call sub6_b_bc(tname)
call sub7_b_bc(tname)
call sub8_b_bc(tname)
call sub9_b_bc(tname)
call sub1_b_c(tname)
call sub2_b_c(tname)
call sub3_b_c(tname)
call sub1_b_gb(tname)
call sub2_b_gb(tname)
call sub9_b_gb(tname)

end program

subroutine sub1_b_bc(tname)
character(*) tname
integer i,irank
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n+1,n+1)
!$xmp template ty(n+1,n+1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j+1,i+1)
!$xmp align b(i,j) with ty(j+1,i+1)

!$xmp loop (i,j) on tx(j+1,i+1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub1_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub2_b_bc(tname)
character(*) tname
integer i,irank
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(0:n,0:n)
!$xmp template ty(0:n,0:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j,i)
!$xmp align b(i,j) with ty(j,i)

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
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub2_b_bc"
call chk_int(tname, ierr)
end subroutine

subroutine sub3_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-2:n,-2:n)
!$xmp template ty(-2:n,-2:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub3_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub4_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-3:n,-3:n)
!$xmp template ty(-3:n,-3:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub4_b_bc"
call chk_int(tname, ierr)
end subroutine

subroutine sub5_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-1:n,-1:n)
!$xmp template ty(-1:n,-1:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub5_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub6_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-4:n,-4:n)
!$xmp template ty(-4:n,-4:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub6_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub7_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-5:n,-5:n)
!$xmp template ty(-5:n,-5:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub7_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub8_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-6:n,-6:n)
!$xmp template ty(-6:n,-6:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub8_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub9_b_bc(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-7:n,-7:n)
!$xmp template ty(-7:n,-7:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub9_b_bc"
call chk_int(tname, ierr)

end subroutine

subroutine sub1_b_c(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(n+1,n+1)
!$xmp template ty(n+1,n+1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(j+1,i+1)
!$xmp align b(i,j) with ty(j+1,i+1)

!$xmp loop (i,j) on tx(j+1,i+1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub1_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine sub2_b_c(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(0:n,0:n)
!$xmp template ty(0:n,0:n)
!$xmp distribute tx(block,block) onto p
!!$xmp distribute ty(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(j,i)
!$xmp align b(i,j) with ty(j,i)

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
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub2_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine sub3_b_c(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
!$xmp nodes p(2,2)
!$xmp template tx(-2:n,-2:n)
!$xmp template ty(-2:n,-2:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub3_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine sub1_b_gb(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,3/)
!$xmp nodes p(2,2)
!$xmp template tx(n+1,n+1)
!$xmp template ty(n+1,n+1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(gblock(m),gblock(m)) onto p
!$xmp align a(i,j) with tx(j+1,i+1)
!$xmp align b(i,j) with ty(j+1,i+1)

!$xmp loop (i,j) on tx(j+1,i+1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j+1,i+1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub1_b_gb"
call chk_int(tname, ierr)

end subroutine

subroutine sub2_b_gb(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/2,3/)
!$xmp nodes p(2,2)
!$xmp template tx(0:n,0:n)
!$xmp template ty(0:n,0:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(gblock(m),gblock(m)) onto p
!$xmp align a(i,j) with tx(j,i)
!$xmp align b(i,j) with ty(j,i)

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
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j,i)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub2_b_gb"
call chk_int(tname, ierr)

end subroutine

subroutine sub9_b_gb(tname)
character(*) tname
integer,parameter :: n=4
integer a(n,n),b(n,n)
integer m(2)=(/5,7/)
!$xmp nodes p(2,2)
!$xmp template tx(-7:n,-7:n)
!$xmp template ty(-7:n,-7:n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(gblock(m),gblock(m)) onto p
!$xmp align a(i,j) with tx(j-1,i-1)
!$xmp align b(i,j) with ty(j-1,i-1)

irank=xmp_node_num()
!$xmp loop (i,j) on tx(j-1,i-1)
do j=1,n
  do i=1,n
    a(i,j)=i+j
  end do
end do

!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=1,n
    b(i,j)=0
  end do
end do

!$xmp gmove
b(2:n,:)=a(2:n,:)

ierr=0
!$xmp loop (i,j) on ty(j-1,i-1)
do j=1,n
  do i=2,n
    ierr=ierr+abs(b(i,j)-(i+j))
  end do
end do

!$xmp reduction (max:ierr)
tname="sub9_b_gb"
call chk_int(tname, ierr)

end subroutine
