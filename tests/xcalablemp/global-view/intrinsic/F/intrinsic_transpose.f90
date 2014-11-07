program xmp_tr

!$xmp nodes p(4)

call test_tr_a2a_bb_c16_1()
!call test_tr_a2a_bb_c8_1()
call test_tr_a2a_bb_i4_1()
call test_tr_a2a_bb_r4_1()
call test_tr_a2a_bb_r8_1()
call test_tr_a2a_bc_i4_1()
call test_tr_a2a_bc_i4_2()
call test_tr_a2a_bc_i4_3()
call test_tr_a2a_bc_i4_4()
call test_tr_bca_bc_i4_1()
call test_tr_bca_bc_i4_2()
call test_tr_bca_bc_i4_3()
call test_tr_bca_bc_i4_4()
call test_tr_cp0_bc_c16_1()
!call test_tr_cp0_bc_c8_1()
call test_tr_cp0_bc_i4_1()
call test_tr_cp0_bc_i4_2()
call test_tr_cp0_bc_r4_1()
call test_tr_cp0_bc_r8_1()
call test_tr_cp0_bg_i4_1()
call test_tr_cp_bc_i4_1()
call test_tr_cp_bc_i4_2()
call test_tr_cp_bc_i4_3()
call test_tr_cp_bc_i4_4()
call test_tr_cps_bc_i4_1()
call test_tr_cps_bc_i4_2()
call test_tr_cps_bc_i4_3()
call test_tr_cps_bc_i4_4()
call test_tr_lc_bb_i4_1()
call test_tr_lc_bb_i4_2()
call test_tr_proj_bc_i4_1()
call test_tr_proj_bc_i4_2()
call test_tr_proj_bc_i4_3()
call test_tr_proj_bc_i4_4()

end program

subroutine test_tr_a2a_bb_c16_1()

integer,parameter :: m=21, n=25
integer ierr
complex*16 a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bb_c8_1()

integer,parameter :: m=21, n=25
integer ierr
complex a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)
end subroutine


subroutine test_tr_a2a_bb_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bb_r4_1()

integer,parameter :: m=21, n=25
integer ierr
real(4) a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bb_r8_1()

integer,parameter :: m=21, n=25
integer ierr
real(8) a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(m)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,*) with ty(i)

  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_a2a_bc_i4_3()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(*,j) with ty(*,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_a2a_bc_i4_4()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,*) with ty(i,*)

  do j=1,n
!$xmp loop (i) on tx(i,*)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_bca_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(m,n)
!$xmp template ty(m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_bca_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(m,n)
!$xmp template ty(n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on q
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_bca_bc_i4_3()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(*,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on q(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_bca_bc_i4_4()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i,*)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on q(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_cp0_bc_c16_1()

integer,parameter :: m=21, n=25
integer ierr
complex*16 a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bc_c8_1()

integer,parameter :: m=21, n=25
integer ierr
complex a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(n+1,m+1)
!$xmp template ty(m+1,n+1)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(j+1,i+1)
!$xmp align b(i,j) with ty(j+1,i+1)

!$xmp loop (i,j) on tx(j+1,i+1)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(j+1,i+1)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(j+1,i+1)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bc_r4_1()

integer,parameter :: m=21, n=25
integer ierr
real a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bc_r8_1()

integer,parameter :: m=21, n=25
integer ierr
real(8) a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp0_bg_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer m1(2)=(/11,14/),m2(2)=(/10,11/)
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(gblock(m1),gblock(m2)) onto p
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(m)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,j) with ty(j)

  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,*) with ty(i)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cp_bc_i4_3()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(*,j) with ty(*,j)

  do j=1,n
!$xmp loop (i) on tx(i,*)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(*,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on p(1,1:2)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_cp_bc_i4_4()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,*) with ty(i,*)

!$xmp loop (j) on tx(*,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i,*)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

!$xmp task on p(1:2,1)
  call chk_int2(ierr)
!$xmp end task

end subroutine


subroutine test_tr_cps_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(4)
!$xmp template tx(m,n)
!$xmp template ty(n)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(i,*) with ty(i)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cps_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(4)
!$xmp template tx(m,n)
!$xmp template ty(m)
!$xmp distribute tx(block,block) onto p
!$xmp distribute ty(cyclic) onto q
!$xmp align a(i,j) with tx(i,j)
!$xmp align b(*,j) with ty(j)

!$xmp loop (i,j) on tx(i,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cps_bc_i4_3()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(4)
!$xmp nodes q(2,2)
!$xmp template tx(m)
!$xmp template ty(n,m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic,cyclic) onto q
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)

  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_cps_bc_i4_4()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(4)
!$xmp nodes q(2,2)
!$xmp template tx(n)
!$xmp template ty(n,m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(cyclic,cyclic) onto q
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_lc_bb_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(m)
!$xmp template ty(m)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(*,j) with ty(j)

  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (j) on ty(j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_lc_bb_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(*)
!$xmp template tx(n)
!$xmp template ty(n)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,*) with ty(i)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
  do j=1,m
!$xmp loop (i) on ty(i)
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_proj_bc_i4_1()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(n)
!$xmp template ty(n,m)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(*,j) with tx(j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (j) on tx(j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_proj_bc_i4_2()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2)=p(1:2,1)
!$xmp template tx(m)
!$xmp template ty(n,m)
!$xmp distribute tx(block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,*) with tx(i)
!$xmp align b(i,j) with ty(i,j)

  do j=1,n
!$xmp loop (i) on tx(i)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_proj_bc_i4_3()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(*,j) with tx(*,j)
!$xmp align b(i,j) with ty(i,j)

!$xmp loop (j) on tx(*,j)
  do j=1,n
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine


subroutine test_tr_proj_bc_i4_4()

integer,parameter :: m=21, n=25
integer ierr
integer a(m,n),b(n,m)
!$xmp nodes p(2,2)
!$xmp nodes q(2,2)
!$xmp template tx(m,n)
!$xmp template ty(n,m)
!$xmp distribute tx(block,block) onto q
!$xmp distribute ty(cyclic,cyclic) onto p
!$xmp align a(i,*) with tx(i,*)
!$xmp align b(i,j) with ty(i,j)

  do j=1,n
!$xmp loop (i) on tx(i,*)
    do i=1,m
      a(i,j)=(i-1)*m+j
    end do
  end do

!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      b(i,j)=0
    end do
  end do

  call xmp_transpose(b, a, 1)

  ierr=0
!$xmp loop (i,j) on ty(i,j)
  do j=1,m
    do i=1,n
      ierr=ierr+abs(b(i,j)-((j-1)*m+i))
    end do
  end do

  call chk_int(ierr)

end subroutine
