program gmove_lc_8

  character*25 tname
  call gmove_lc_32a3t_b2(tname)
  call gmove_lc_32a3t_b(tname)
  call gmove_lc_3a3t_bc(tname)
  call gmove_lc_3a3t_b(tname)
  call gmove_lc_3a3t_b_h(tname)
  call gmove_lc_3a3t_c(tname)
  call gmove_lc_3a3t_c_h(tname)
  call gmove_lc_3a3t_gb(tname)

end program

subroutine gmove_lc_32a3t_b2(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1)=a(1,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,1
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-i-j-1)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_32a3t_b2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_32a3t_b(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1,1:n,1:n)=a(1:n,1,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,1
      ierr=ierr+abs(b(i,j,k)-1-j-k)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_32a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_b(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_b_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_b_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_c(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_c_h(tname)

character(*) tname
integer i
integer,parameter :: n=9
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-a(i,j,k))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_c_h"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_lc_3a3t_gb(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer mx(2)=(/2,6/)
integer my(2)=(/2,6/)
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(gblock(mx),gblock(mx),gblock(mx)) onto p
!$xmp distribute ty(gblock(my),gblock(my),gblock(my)) onto p
!$xmp align a(i,j,k) with tx(i,j,k)
!$xmp align b(i,j,k) with ty(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

!$xmp loop (i,j,k) on ty(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      b(i,j,k)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i,j,k) on ty(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      ierr=ierr+abs(b(i,j,k)-i-j-k)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_lc_3a3t_gb"
call chk_int(tname, ierr)

end subroutine
