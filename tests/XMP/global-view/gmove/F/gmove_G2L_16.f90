program gmove_G2L_16

  character*25 tname
  call gmove_G2L_1a4t_bc(tname)
  call gmove_G2L_4a4t_bc(tname)

end program

subroutine gmove_G2L_1a4t_bc(tname)

character(*) tname
integer,parameter :: n=8
integer a(n,n,n,n), b(n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(*,*,*,i3) with tx(*,*,*,i3)

!$xmp loop (i3) on tx(*,*,*,i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        a(i0,i1,i2,i3)=i0+i1+i2+i3
      end do
    end do
  end do
end do

do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        b(i0,i1,i2,i3)=0
      end do
    end do
  end do
end do

!$xmp gmove
b(2:n,2:n,2:n,2:n)=a(2:n,2:n,2:n,2:n)

ierr=0
do i3=2,n
  do i2=2,n
    do i1=2,n
      do i0=2,n
        ierr=ierr+abs(b(i0,i1,i2,i3)-i0-i1-i2-i3)
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_1a4t_bc"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_G2L_4a4t_bc(tname)

character(*) tname
integer i
integer,parameter :: n=8
integer a(n,n,n,n),b(n,n,n,n)
integer xmp_node_num
!$xmp nodes p(2,2,2,2)
!$xmp template tx(n,n,n,n)
!$xmp distribute tx(cyclic(2),cyclic(2),cyclic(2),cyclic(2)) onto p
!$xmp align a(i0,i1,i2,i3) with tx(i0,i1,i2,i3)

!$xmp loop (i0,i1,i2,i3) on tx(i0,i1,i2,i3)
do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        a(i0,i1,i2,i3)=i0+i1+i2+i3
      end do
    end do
  end do
end do

do i3=1,n
  do i2=1,n
    do i1=1,n
      do i0=1,n
        b(i0,i1,i2,i3)=0
      end do
    end do
  end do
end do

!$xmp gmove
b(2:5,2:5,2:5,2:5)=a(5:8,5:8,5:8,5:8)

ierr=0
do i3=2,5
  do i2=2,5
    do i1=2,5
      do i0=2,5
        ierr=ierr+abs(b(i0,i1,i2,i3)-(i0+3)-(i1+3)-(i2+3)-(i3+3))
      end do
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_G2L_4a4t_bc"
call chk_int(tname, ierr)

end subroutine

