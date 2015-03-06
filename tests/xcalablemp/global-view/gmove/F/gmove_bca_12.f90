program gmove_bca_12

  character*25 tname
  call gmove_bca_3a3t_c_b2(tname)
  call gmove_bca_3a3t_c_b2_s(tname)
  call gmove_bca_3a3t_c_gb(tname)
  call gmove_bca_3a3t_c_gb_s(tname)

end program

subroutine gmove_bca_3a3t_c_b2(tname)

character(*) tname
integer,parameter :: n=16
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,3,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp distribute ty(block,block,block) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i2,i1) with ty(*,i1,i2)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i2,i1)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i1)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_bca_3a3t_c_b2"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_c_b2_s(tname)

character(*) tname
integer,parameter :: n=16
integer a(n,n,n),b(n,n,n)
!$xmp nodes p(2,3,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp distribute ty(block,block,block) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i2,i1) with ty(*,i1,i2)
!$xmp shadow b(0,0,1)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i2,i1)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i1)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_bca_3a3t_c_b2_s"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_c_gb(tname)

character(*) tname
integer,parameter :: n=16
integer a(n,n,n),b(n,n,n)
integer m1(2)=(/3,13/),m2(3)=(/6,5,5/)
!$xmp nodes p(2,3,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp distribute ty(gblock(m1),gblock(m2),gblock(m1)) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i2,i1) with ty(*,i1,i2)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i2,i1)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i1)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_bca_3a3t_c_gb"
call chk_int(tname, ierr)

end subroutine

subroutine gmove_bca_3a3t_c_gb_s(tname)

character(*) tname
integer,parameter :: n=16
integer a(n,n,n),b(n,n,n)
integer m1(2)=(/3,13/),m2(3)=(/6,5,5/)
integer xmp_node_num
!$xmp nodes p(2,3,2)
!$xmp template tx(n,n,n)
!$xmp template ty(n,n,n)
!$xmp distribute tx(cyclic,cyclic,cyclic) onto p
!$xmp distribute ty(gblock(m1),gblock(m2),gblock(m1)) onto p
!$xmp align a(i0,i1,i2) with tx(i0,i1,i2)
!$xmp align b(*,i2,i1) with ty(*,i1,i2)
!$xmp shadow b(0,0,1)

!$xmp loop (i0,i1,i2) on tx(i0,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      a(i0,i1,i2)=i0+i1+i2
    end do
  end do
end do

!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      b(i0,i2,i1)=0
    end do
  end do
end do

!$xmp gmove
b(1:n,1:n,1:n)=a(1:n,1:n,1:n)

ierr=0
!$xmp loop (i2,i1) on ty(*,i1,i2)
do i2=1,n
  do i1=1,n
    do i0=1,n
      ierr=ierr+abs(b(i0,i2,i1)-i0-i1-i2)
    end do
  end do
end do

!$xmp reduction (max:ierr)
tname="gmove_bca_3a3t_c_gb_s"
call chk_int(tname, ierr)

end subroutine

