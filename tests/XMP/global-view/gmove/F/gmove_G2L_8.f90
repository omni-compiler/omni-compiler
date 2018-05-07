program gmove_G2L_8

  character*25 tname
  call gmove_G2L_1a3t_bc(tname)
  call gmove_G2L_1a3t_bc2(tname)
  call gmove_G2L_3a3t_bc(tname)

end program


subroutine gmove_G2L_1a3t_bc(tname)

character(*) tname
integer,parameter :: n=8
integer a(n,n,n), b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(*,*,i2) with tx(*,*,i2)

!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

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
do i2=2,n
  do i1=2,n
    do i0=2,n
      ierr=ierr+abs(b(i0,i1,i2)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a3t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_G2L_1a3t_bc2(tname)

character(*) tname
integer,parameter :: n=8
integer a(n,n,n), b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(*,*,i2) with tx(*,*,i2)

!$xmp loop (i2) on tx(*,*,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

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
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i1,i2)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a3t_bc2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_G2L_3a3t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n),b(n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2)
!$xmp template tx(n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i,j,k) with tx(i,j,k)

!$xmp loop (i,j,k) on tx(i,j,k)
do k=1,n
  do j=1,n
    do i=1,n
      a(i,j,k)=i+j+k
    end do
  end do
end do

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
do k=2,5
  do j=2,5
    do i=2,5
      ierr=ierr+abs(b(i,j,k)-(i+3+j+3+k+3))
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_3a3t_bc"
call chk_int(tname, ierr)

end subroutine
