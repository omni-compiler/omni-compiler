program gmove_cp_8

  character*25 tname
  call gmove_cp_3a3t_bc(tname)
  call gmove_cp_3a3t_b_c(tname)
  call gmove_cp_3a3t_b(tname)
  call gmove_cp_3a3t_c(tname)

end program

subroutine gmove_cp_3a3t_bc(tname)

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
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=2,5
  do j=2,5
    do i=2,5
      ierr=ierr+abs(b(i,j,k)-(i+3+j+3+k+3))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_3a3t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_3a3t_b_c(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(block,block,block) onto p
!$xmp distribute ty(cyclic,cyclic,cyclic) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(i0,i1,i2) with ty(i0,i1,i2)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i0,i1,i2) on ty(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i1,i2)=0
    end do
  end do
end do

!$xmp gmove
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i0,i1,i2) on ty(i0,i1,i2)
do i2=2,5
  do i1=2,5
    do i0=2,5
      ierr=ierr+abs(b(i0,i1,i2)-(i0+3)-(i1+3)-(i2+3))
!      print *, 'i0=',i1,'i1=',i1,'i2=',i2,'b=',b(i0,i1,i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_3a3t_b_c"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_3a3t_b(tname)

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
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=2,5
  do j=2,5
    do i=2,5
      ierr=ierr+abs(b(i,j,k)-(i+3+j+3+k+3))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_3a3t_b"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_cp_3a3t_c(tname)

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
b(2:5,2:5,2:5)=a(5:8,5:8,5:8)

ierr=0
!$xmp loop (i,j,k) on tx(i,j,k)
do k=2,5
  do j=2,5
    do i=2,5
      ierr=ierr+abs(b(i,j,k)-(i+3+j+3+k+3))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_cp_3a3t_c"
call chk_int(tname, ierr)

end subroutine

