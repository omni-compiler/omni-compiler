program test_transpose_Fjt

!  call test_transpose_001()
!  call test_transpose_002()
!  call test_transpose_003()
  call test_transpose_004()
  call test_transpose_005()
  call test_transpose_006()
  call test_transpose_007()
  call test_transpose_008()
  call test_transpose_009()
  call test_transpose_010()
!  call test_transpose_011()
!  call test_transpose_012()
  call test_transpose_013()
  call test_transpose_014()
  call test_transpose_015()
!  call test_transpose_016()
  call test_transpose_017()
  call test_transpose_018()
!  call test_transpose_019()
!  call test_transpose_020()
  call test_transpose_021()
  call test_transpose_022()
!  call test_transpose_023()
  call test_transpose_030()
  call test_transpose_031()
  call test_transpose_032()
  call test_transpose_033()
  call test_transpose_034()
  call test_transpose_035()
  call test_transpose_036()
  call test_transpose_037()
  call test_transpose_038()
  call test_transpose_039()
  call test_transpose_040()
  call test_transpose_041()
  call test_transpose_042()
  call test_transpose_043()
  call test_transpose_044()
  call test_transpose_045()
  call test_transpose_046()
  call test_transpose_047()
  call test_transpose_048()
  call test_transpose_049()
  call test_transpose_050()
  call test_transpose_051()
  call test_transpose_052()
  call test_transpose_053()
  call test_transpose_101()
  call test_transpose_102()
  call test_transpose_104()
  call test_transpose_105()
  call test_transpose_106()
!  call test_transpose_107()
  call test_transpose_108()

end program

!subroutine test_transpose_001()
!
!  integer a(8,16), b(16,8)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes p(10)=q(1:10)
!!$xmp nodes pa(3,3)=p(1:9)
!!$xmp nodes pb(4,2)=p(3:10)
!!$xmp template ta(8,16)
!!$xmp template tb(16,8)
!!$xmp distribute ta(cyclic(3),cyclic) onto pa
!!$xmp distribute tb(block,gblock((/2,6/))) onto pb
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!!$xmp shadow a(1,0)
!!$xmp shadow b(1,2)
!
!  b = -2
!  a = -1
!!$xmp loop (i,j) on ta(i,j)
!  do j=1, 16
!   do i=1, 8
!     a(i,j) = -1*((j-1)*8+i)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i,j) on tb(i,j) reduction(+: error)
!  do j=1, 8
!     do i=1, 16
!        if(b(i,j) .ne.  -1*((i-1)*8+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 1)
!
!end subroutine

!subroutine test_transpose_002()
!
!  integer a(8,16), b(16,8)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes pa(2,4)=q(1:8)
!!$xmp nodes pb(4,2)=1(1:8)
!!$xmp template ta(8,16)
!!$xmp template tb(16,8)
!!$xmp distribute ta(block,cyclic(3)) onto pa
!!$xmp distribute tb(block,gblock((/2,6/))) onto pb
!!!$xmp distribute tb(cyclic,cyclic) onto pb
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!!$xmp shadow a(1,0)
!!$xmp shadow b(1,2)
!
!  b = -2
!!$xmp loop (i,j) on tb(i,j)
!  do j=1, 8
!   do i=1, 16
!     b(i,j) = (j-1)*16+i
!   enddo
!  enddo
!!!$xmp reflect (b)
!!!$xmp loop (i,j) on tb(i,j)
!!  do j=1, 8
!!   do i=1, 16
!!     b(i,j) = -1
!!   enddo
!!  enddo
!
!  call xmp_transpose(a, b, 0)
!
!  error = 0
!!$xmp loop (i,j) on tb(i,j) reduction(+: error)
!  do j=1, 8
!     do i=1, 16
!        if(b(i,j) .ne. (j-1)*16+i) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 1)
!
!end subroutine

!subroutine test_transpose_003()
!
!  integer a(8,16), b(16,8)
!  integer m(2)=(/2,6/)
!!$xmp nodes q(16)
!!$xmp nodes p(8)=q(1:8)
!!$xmp nodes pa(2,4)=q(1:8)
!!$xmp nodes pb(4,2)=q(1:8)
!!$xmp template ta(8,16)
!!$xmp template tb(16,8)
!!$xmp distribute ta(cyclic(3),cyclic) onto pa
!!!$xmp distribute ta(block,block) onto pa
!!$xmp distribute tb(block,gblock((/2,6/))) onto pb
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!!$xmp shadow a(1,0)
!!$xmp shadow b(1,2)
!
!  b = -2
!  a = -1
!!!$xmp loop (i,j) on tb(i,j)
!!  do j=1, 8
!!   do i=1, 16
!!     b(i,j) = (j-1)*16+i
!!   enddo
!!  enddo
!!$xmp loop (i,j) on ta(i,j)
!  do j=1, 16
!   do i=1, 8
!     a(i,j) = ((j-1)*8+i)
!   enddo
!  enddo
!!!$xmp reflect (b)
!!!$xmp loop (i,j) on tb(i,j)
!!  do j=1, 8
!!   do i=1, 16
!!     b(i,j) = -1
!!   enddo
!!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i,j) on tb(i,j) reduction(+: error)
!  do j=1, 8
!     do i=1, 16
!        if(b(i,j) .ne.  ((i-1)*8+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 1)
!
!end subroutine

subroutine test_transpose_004()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(block) onto p
!$xmp align a(*,j) with t(j)
!$xmp align b(*,j) with t(j)

  b = -2
  a = -1
!$xmp loop (j) on t(j)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on t(j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_005()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(20)
!$xmp distribute t(block) onto p
!$xmp align a(j,*) with t(j+1)
!$xmp align b(*,j) with t(j+1)
!$xmp shadow a(2,0)
!$xmp shadow b(0,1)

  b = -2
  a = -1
!$xmp loop (i) on t(i+1)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on t(j+1) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_006()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(block) onto p
!$xmp align a(*,j) with t(j)
!$xmp align b(j,*) with t(j)

  b = -2
  a = -1
!$xmp loop (j) on t(j)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on t(i) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_007()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(cyclic) onto p
!$xmp align a(j,*) with t(j)
!$xmp align b(*,j) with t(j)

  b = -2
  a = -1
!$xmp loop (i) on t(i)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on t(j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_008()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(cyclic) onto p
!$xmp align a(*,j) with t(j)
!$xmp align b(j,*) with t(j)

  b = -2
  a = -1
!$xmp loop (j) on t(j)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on t(i) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_009()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(cyclic(3)) onto p
!$xmp align a(j,*) with t(j)
!$xmp align b(*,j) with t(j)

  b = -2
  a = -1
!$xmp loop (i) on t(i)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on t(j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_010()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template t(16)
!$xmp distribute t(cyclic(3)) onto p
!$xmp align a(*,j) with t(j)
!$xmp align b(j,*) with t(j)

  b = -2
  a = -1
!$xmp loop (j) on t(j)
  do j=1, 16
   do i=1, 16
     a(i,j) = (j-1)*16+i
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on t(i) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

!subroutine test_transpose_011()
!
!  integer a(16,16), b(16,16)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes p(4)=q(1:4)
!!$xmp template t(16)
!!$xmp distribute t(gblock((/2,3,4,7/))) onto p
!!$xmp align a(j,*) with t(j)
!!$xmp align b(*,j) with t(j)
!
!  b = -2
!  a = -1
!!$xmp loop (i) on t(i)
!  do j=1, 16
!   do i=1, 16
!     a(i,j) = (j-1)*16+i
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (j) on t(j) reduction(+: error)
!  do j=1, 16
!     do i=1, 16
!        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 1)
!
!end subroutine

!subroutine test_transpose_012()
!
!  integer a(16,16), b(16,16)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes p(4)=q(1:4)
!!$xmp template t(16)
!!$xmp distribute t(gblock((/7,4,3,2/))) onto p
!!$xmp align a(*,j) with t(j)
!!$xmp align b(j,*) with t(j)
!!$xmp shadow a(0,3)
!
!  b = -2
!  a = -1
!!$xmp loop (j) on t(j)
!  do j=1, 16
!   do i=1, 16
!     a(i,j) = (j-1)*16+i
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i) on t(i) reduction(+: error)
!  do j=1, 16
!     do i=1, 16
!        if(b(i,j) .ne.  (i-1)*16+j) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error,1)
!
!end subroutine

subroutine test_transpose_013()

  integer,parameter:: M=23, N=31
  integer a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template ta(N)
!$xmp template tb(M)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(block) onto p
!$xmp align a(*,i) with ta(i)
!$xmp align b(*,i) with tb(i)

  b = -2
  a = -1
!$xmp loop (j) on ta(j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_014()

  integer,parameter:: M=23, N=31
  integer*8 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template ta(N)
!$xmp template tb(M)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(cyclic) onto p
!$xmp align a(i,*) with ta(i)
!$xmp align b(*,i) with tb(i)
!$xmp shadow a(2,0)

  b = -2
  a = -1
!$xmp loop (i) on ta(i)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_015()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template ta(N)
!$xmp template tb(N)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(cyclic(3)) onto p
!$xmp align a(*,i) with ta(i)
!$xmp align b(i,*) with tb(i)
!$xmp shadow a(0,2)

  b = -2
  a = -1
!$xmp loop (j) on ta(j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on tb(i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

!subroutine test_transpose_016()
!
!  integer,parameter:: M=23, N=31
!  real*4 a(M,N), b(N,M)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes p(4)=q(1:4)
!!$xmp template ta(M)
!!$xmp template tb(N)
!!$xmp distribute ta(block) onto p
!!$xmp distribute tb(gblock((/6,9,7,9/))) onto p
!!$xmp align a(i,*) with ta(i)
!!$xmp align b(i,*) with tb(i)
!!$xmp shadow a(1,0)
!!$xmp shadow b(2,0)
!
!  b = -2
!  a = -1
!!$xmp loop (i) on ta(i)
!  do j=1, N
!   do i=1, M
!     a(i,j) = -1*((j-1)*M+i)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i) on tb(i) reduction(+: error)
!  do j=1, M
!     do i=1, N
!        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 1)
!
!end subroutine

subroutine test_transpose_017()

  integer,parameter:: M=23, N=31
  real*8 a(M,N), b(N,M)
  integer error
!$xmp nodes p0(16)
!$xmp nodes p(8)=p0(1:8)
!$xmp nodes q(2,2)=p(1:4)
!$xmp template ta(M,N)
!$xmp template tb(M)
!$xmp distribute ta(cyclic,block) onto q
!$xmp distribute tb(block) onto p
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(*,i) with tb(i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_018()

  integer,parameter:: M=23, N=31
!  complex*8 a(M,N), b(N,M)  complex*8 is error
  complex*16 a(M,N), b(N,M)
  integer error
!$xmp nodes p0(16)
!$xmp nodes p(8)=p0(1:8)
!$xmp nodes q(2,2)=p(2:5)
!$xmp template ta(M,N)
!$xmp template tb(N,M)
!$xmp distribute ta(cyclic(2),cyclic(3)) onto q
!$xmp distribute tb(block,block) onto q
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i,j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

!subroutine test_transpose_019()
!
!  integer,parameter:: M=23, N=31
!  complex*16 a(M,N), b(N,M)
!  integer error
!!$xmp nodes p0(16)
!!$xmp nodes p(8)=p0(1:8)
!!$xmp nodes q(2,2)=p(2:5)
!!$xmp template ta(M,N)
!!$xmp template tb(N,M)
!!$xmp distribute ta(block,block) onto q
!!$xmp distribute tb(gblock((/20,11/)),gblock((/10,13/))) onto q
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!$xmp shadow a(1,2)
!!$xmp shadow b(2,1)
!
!  b = -2
!  a = -1
!!$xmp loop (i,j) on ta(i,j)
!  do j=1, N
!   do i=1, M
!     a(i,j) = -1*((j-1)*M+i)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i,j) on tb(i,j) reduction(+: error)
!  do j=1, M
!     do i=1, N
!        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 2)
!
!end subroutine

!subroutine test_transpose_020()
!
!  integer,parameter:: M=23, N=31
!  integer*4 a(M,N), b(N,M)
!  integer error
!!$xmp nodes p0(16)
!!$xmp nodes p(8)=p0(1:8)
!!$xmp nodes q(2,2)=p(2:5)
!!$xmp template ta(M,N)
!!$xmp template tb(N,M)
!!$xmp distribute ta(cyclic,cyclic(3)) onto q
!!$xmp distribute tb(gblock((/20,11/)),gblock((/10,13/))) onto q
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!$xmp shadow b(2,1)
!
!  b = -2
!  a = -1
!!$xmp loop (i,j) on ta(i,j)
!  do j=1, N
!   do i=1, M
!     a(i,j) = -1*((j-1)*M+i)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i,j) on tb(i,j) reduction(+: error)
!  do j=1, M
!     do i=1, N
!        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 2)
!
!end subroutine

subroutine test_transpose_021()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template ta(N+5)
!$xmp template tb(M+7)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(block) onto p
!$xmp align a(*,j) with ta(j+2)
!$xmp align b(*,j) with tb(j+4)

  b = -2
  a = -1
!$xmp loop (j) on ta(j+2)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j+4) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_022()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes p(4)=q(1:4)
!$xmp template ta(M+5)
!$xmp template tb(M+7)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(cyclic) onto p
!$xmp align a(j,*) with ta(j+2)
!$xmp align b(*,j) with tb(j+4)
!$xmp shadow a(0,2)

  b = -2
  a = -1
!$xmp loop (i) on ta(i+2)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j+4) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

!subroutine test_transpose_023()
!
!  integer,parameter:: M=23, N=31
!  integer*8 a(M,N), b(N,M)
!  integer error
!!$xmp nodes q(16)
!!$xmp nodes p(4)=q(1:4)
!!$xmp template ta(N+5)
!!$xmp template tb(N+7)
!!$xmp distribute ta(cyclic(4)) onto p
!!$xmp distribute tb(gblock((/8,10,7,13/))) onto p
!!$xmp align a(*,j) with ta(j+3)
!!$xmp align b(j,*) with tb(j+7)
!!$xmp shadow b(2,0)
!
!  b = -2
!  a = -1
!!$xmp loop (j) on ta(j+3)
!  do j=1, N
!   do i=1, M
!     a(i,j) = -1*((j-1)*M+i)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!
!  error = 0
!!$xmp loop (i) on tb(i+7) reduction(+: error)
!  do j=1, M
!     do i=1, N
!        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int3(error, 2)
!
!end subroutine

subroutine test_transpose_030()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(M,M,M)
!$xmp distribute ta(block,block) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp align a(*,j) with ta(*,j)
!$xmp align b(*,j) with tb(j,*,*)

  b = -2
  a = -1
!$xmp loop (j) on ta(*,j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j,*,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_031()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N)
!$xmp template tb(M,M,M)
!$xmp distribute ta(cyclic(3)) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp align a(*,j) with ta(j)
!$xmp align b(*,j) with tb(j,*,*)

  b = -2
  a = -1
!$xmp loop (j) on ta(j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(j,*,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_032()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,10/)
  integer m2(2)=(/10,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N+5)
!$xmp template tb(M,M,M)
!$xmp distribute ta(cyclic(3)) onto pa
!$xmp distribute tb(gblock(m1),gblock(m2),gblock(m3)) onto pb
!$xmp align a(*,j) with ta(j+2)
!$xmp align b(*,j) with tb(*,j,*)

  b = -2
  a = -1
!$xmp loop (j) on ta(j+2)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(*,j,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_033()

  integer,parameter:: M=23, N=31
  real*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,10/)
  integer m2(2)=(/10,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M)
!$xmp template tb(M,M,M)
!$xmp distribute ta(block) onto pa
!$xmp distribute tb(gblock(m1),gblock(m2),gblock(m3)) onto pb
!$xmp align a(i,*) with ta(i)
!$xmp align b(*,j) with tb(*,j,*)
!$xmp shadow a(2,0)
!$xmp shadow b(0,2)

  b = -2
  a = -1
!$xmp loop (i) on ta(i)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (j) on tb(*,j,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_034()

  integer,parameter:: M=23, N=31
  real*8 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M)
!$xmp template tb(N,M,M)
!$xmp distribute ta(block) onto pa
!$xmp distribute tb(block,block,block) onto pb
!$xmp align a(i,*) with ta(i)
!$xmp align b(i,j) with tb(i,*,j)

  b = -2
  a = -1
!$xmp loop (i) on ta(i)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i,*,j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne.  -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 2)

end subroutine

subroutine test_transpose_035()

  integer,parameter:: M=23, N=31
  complex*16 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M+5,N+7)
!$xmp template tb(N+5,M,M+7)
!$xmp distribute ta(block,cyclic) onto pa
!$xmp distribute tb(block,block,block) onto pb
!$xmp align a(i,j) with ta(i+2,j+3)
!$xmp align b(i,j) with tb(i+2,*,j+3)
!$xmp shadow b(1,1)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i+2,j+3)
  do j=1, N
   do i=1, M
     a(i,j) = dcmplx(-1*((j-1)*M+i),0)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i+2,*,j+3) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. dcmplx(-1*((i-1)*M+j))) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_036()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(N,N,M)
!$xmp distribute ta(block,cyclic) onto pa
!$xmp distribute tb(block,block,cyclic(4)) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,i,j)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,i,j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_037()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,18/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(N,N,M)
!$xmp distribute ta(block,gblock(m1)) onto pa
!$xmp distribute tb(block,block,cyclic(3)) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,i,j)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,i,j) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_038()

  integer,parameter:: M=23, N=31
  integer*8 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,23/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(8)=q(1:8)
!$xmp template ta(M+3,N+5)
!$xmp template tb(N+5)
!$xmp distribute ta(block,gblock(m1)) onto pa
!$xmp distribute tb(cyclic) onto pb
!$xmp align a(i,j) with ta(i+2,j+2)
!$xmp align b(i,*) with tb(i+3)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i+2,j+2)
  do j=1, N
   do i=1, M
     a(i,j) = -1*((j-1)*M+i)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on tb(i+3) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_039()

  integer,parameter:: M=23, N=31
  real*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(8)=q(1:8)
!$xmp template ta(N,M)
!$xmp template tb(N)
!$xmp distribute ta(cyclic,block) onto pa
!$xmp distribute tb(cyclic) onto pb
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,*) with tb(i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j,i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on tb(i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_040()

  integer,parameter:: M=23, N=31
  real*8 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N,M)
!$xmp template tb(M,N,N)
!$xmp distribute ta(cyclic,block) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,j) with tb(j,i,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j,i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(j,i,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_041()

  integer,parameter:: M=23, N=31
  complex*16 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/13,20/)
  integer m2(2)=(/20,13/)
  integer m3(2)=(/11,12/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N+3)
!$xmp template tb(M+2,N,N)
!$xmp distribute ta(cyclic,cyclic(3)) onto pa
!$xmp distribute tb(cyclic,cyclic,cyclic) onto pb
!$xmp align a(i,j) with ta(i,j+3)
!$xmp align b(i,j) with tb(j+2,i,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j+3)
  do j=1, N
     do i=1, M
        a(i,j) = dcmplx(-1*((j-1)*M+i),((j-1)*M+i))
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(j+2,i,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. dcmplx(-1*((i-1)*M+j),(i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_042()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/5,26/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(M,M,N)
!$xmp distribute ta(cyclic,cyclic(3)) onto pa
!$xmp distribute tb(cyclic,cyclic,gblock(m1)) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(*,j,i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,j,i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_043()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/26,5/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M)
!$xmp template tb(M,M,N)
!$xmp distribute ta(cyclic(4)) onto pa
!$xmp distribute tb(cyclic,cyclic,gblock(m1)) onto pb
!$xmp align a(i,*) with ta(i)
!$xmp align b(i,j) with tb(*,j,i)

  b = -2
  a = -1
!$xmp loop (i) on ta(i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,j,i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_044()

  integer,parameter:: M=23, N=31
  integer*8 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/26,5/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M+5)
!$xmp template tb(M,M,N+1)
!$xmp distribute ta(cyclic(4)) onto pa
!$xmp distribute tb(cyclic,block,block) onto pb
!$xmp align a(i,*) with ta(i+1)
!$xmp align b(i,j) with tb(j,*,i+1)

  b = -2
  a = -1
!$xmp loop (i) on ta(i+1)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(j,*,i+1) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_045()

  integer,parameter:: M=23, N=31
  real*4 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/26,5/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(M,M,N)
!$xmp distribute ta(cyclic(5),cyclic) onto pa
!$xmp distribute tb(cyclic(3),block,block) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(j,*,i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(j,*,i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_046()

  integer,parameter:: M=23, N=31
  real*8 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/26,5/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(N,M,M)
!$xmp distribute ta(cyclic(5),cyclic) onto pa
!$xmp distribute tb(cyclic(2),cyclic(3),cyclic(4)) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i,j,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_047()

  integer,parameter:: M=23, N=31
  complex*16 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/1,22/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N+3,M)
!$xmp template tb(N,M+2,M)
!$xmp distribute ta(cyclic(5),gblock(m1)) onto pa
!$xmp distribute tb(cyclic(2),cyclic(3),cyclic(4)) onto pb
!$xmp align a(i,j) with ta(j+3,i)
!$xmp align b(i,j) with tb(i,j+1,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j+3,i)
  do j=1, N
     do i=1, M
        a(i,j) = dcmplx(-1*((j-1)*M+i),(j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i,j+1,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. dcmplx(-1*((i-1)*M+j),(i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_048()

  integer,parameter:: M=23, N=31
  integer*2 a(M,N), b(N,M)
  integer error
  integer m1(2)=(/23,0/)
  integer m2(8)=(/2,7,3,2,6,1,4,6/)
!$xmp nodes q(16)
!$xmp nodes pa(2,4)=q(1:8)
!$xmp nodes pb(8)=q(1:8)
!$xmp template ta(N,M)
!$xmp template tb(N)
!$xmp distribute ta(cyclic(2),gblock(m1)) onto pa
!$xmp distribute tb(gblock(m2)) onto pb
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,*) with tb(i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j,i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on tb(i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_049()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
  integer m1(4)=(/4,5,6,8/)
  integer m2(8)=(/2,7,3,2,6,1,4,6/)
!$xmp nodes q(16)
!$xmp nodes pa(4,2)=q(1:8)
!$xmp nodes pb(8)=q(1:8)
!$xmp template ta(M,N)
!$xmp template tb(N)
!$xmp distribute ta(gblock(m1),block) onto pa
!$xmp distribute tb(gblock(m2)) onto pb
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,*) with tb(i)
!$xmp shadow a(2,3)
!$xmp shadow b(1,0)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i,j)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i) on tb(i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_050()

  integer,parameter:: M=23, N=31
  integer*8 a(M,N), b(N,M)
  integer error
  integer m1(4)=(/10,4,5,6/)
  integer m2(2)=(/32,1/) ! error
!  integer m2(2)=(/31,2/)
!$xmp nodes q(16)
!$xmp nodes pa(4,2)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(M+2,N+2)
!$xmp template tb(N+2,M+3,M)
!$xmp distribute ta(gblock(m1),block) onto pa
!$xmp distribute tb(gblock(m2),cyclic,block) onto pb
!$xmp align a(i,j) with ta(i+1,j+2)
!$xmp align b(i,j) with tb(i+2,j+3,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(i+1,j+2)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i+2,j+3,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_051()

  integer,parameter:: M=23, N=31
  real*4 a(M,N), b(N,M)
  integer error
  integer m1(4)=(/2,5,8,16/)
  integer m2(2)=(/1,30/)
!$xmp nodes q(16)
!$xmp nodes pa(4,2)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N,M)
!$xmp template tb(N,M,M)
!$xmp distribute ta(gblock(m1),cyclic(4)) onto pa
!$xmp distribute tb(gblock(m2),cyclic,block) onto pb
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,j) with tb(i,j,*)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j,i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(i,j,*) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_052()

  integer,parameter:: M=23, N=31
  real*8 a(M,N), b(N,M)
  integer error
  integer m1(4)=(/2,5,8,16/)
  integer m2(2)=(/12,11/)
  integer m3(2)=(/21,10/)
!$xmp nodes q(16)
!$xmp nodes pa(4,2)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N,M)
!$xmp template tb(M,M,N)
!$xmp distribute ta(gblock(m1),cyclic(4)) onto pa
!$xmp distribute tb(cyclic(3),gblock(m2),gblock(m3)) onto pb
!$xmp align a(i,j) with ta(j,i)
!$xmp align b(i,j) with tb(*,j,i)

  b = -2
  a = -1
!$xmp loop (i,j) on ta(j,i)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,j,i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_053()

  integer,parameter:: M=23, N=31
  complex*16 a(M,N), b(N,M)
  integer error
  integer m2(2)=(/16,21/)
  integer m3(2)=(/21,10/)
!$xmp nodes q(16)
!$xmp nodes pa(8)=q(1:8)
!$xmp nodes pb(2,2,2)=q(1:8)
!$xmp template ta(N)
!$xmp template tb(M,M+4,N)
!$xmp distribute ta(block) onto pa
!$xmp distribute tb(block,gblock(m2),gblock(m3)) onto pb
!$xmp align a(*,j) with ta(j)
!$xmp align b(i,j) with tb(*,j+2,i)
!$xmp shadow a(0,2)
!$xmp shadow b(2,2)

  b = -2
  a = -1
!$xmp loop (j) on ta(j)
  do j=1, N
     do i=1, M
        a(i,j) = -1*((j-1)*M+i)
     enddo
  enddo

  call xmp_transpose(b, a, 0)

  error = 0
!$xmp loop (i,j) on tb(*,j+2,i) reduction(+: error)
  do j=1, M
     do i=1, N
        if(b(i,j) .ne. -1*((i-1)*M+j)) error = error+1
     enddo
  enddo

  call chk_int3(error, 1)

end subroutine

subroutine test_transpose_101()

  integer a(4,4), b(4,4), rc(4,4)
  integer err
!$xmp nodes p(16)
!$xmp nodes pa(4,4)=p(1:16)
!$xmp template ta(4,4)
!$xmp distribute ta(block,block) onto pa
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)

  b = -2
  a = -1
  err = 0
!$xmp loop (i,j) on ta(i,j)
  do j=1, 4
   do i=1, 4
     a(i,j) = i*(j-1)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

!$xmp gmove
  rc(:,:) = b(:,:)

  do j=1, 4
   do i=1, 4
      if (rc(i,j). ne. (i-1)*j ) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine

subroutine test_transpose_102()

  real*4 a(4,4), b(4,4), rc(4,4)
  integer err
!$xmp nodes p(16)
!$xmp nodes pa(4,4)=p(1:16)
!$xmp template ta(4,4)
!$xmp distribute ta(block,block) onto pa
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)

  b = -2.0e0
  a = -1.0e0
  rc = -3.0e0
  err = 0
!$xmp loop (i,j) on ta(i,j)
  do j=1, 4
   do i=1, 4
     a(i,j) = real(i)*real(j-1)
   enddo
  enddo

  call xmp_transpose(b, a, 0)

!$xmp gmove
  rc(:,:) = b(:,:)

  do j=1, 4
   do i=1, 4
      if (rc(i,j). ne. real(i-1)*real(j) ) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine

subroutine test_transpose_104()

  logical a(4,4), b(4,4), rc(4,4), rd(4,4)
  integer err
!$xmp nodes p(16)
!$xmp nodes pa(4,4)=p(1:16)
!$xmp template ta(4,4)
!$xmp distribute ta(block,block) onto pa
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)

  b = .false.
  a = .true.
  rc = .true.
  rd = .true.

  err = 0
!$xmp loop (i,j) on ta(i,j)
  do j=1, 4
   do i=1, 4
      if ( mod(j,2) .eq. 0) then
         a(i,j) = .false.
      endif
   enddo
  enddo


 call xmp_transpose(b, a, 0)

!$xmp loop (i,j) on ta(i,j) reduction(+:err)
  do j=1, 4
   do i=1, 4
      if ( (mod(i,2).eq.0) .and. ( b(i,j) .eqv. .true.) ) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine

subroutine test_transpose_105()

  integer a(4,4), b(4,4), c(4,4), rc(4,4)
  integer err
!$xmp nodes p(16)
!$xmp nodes pa(4,4)=p(1:16)
!$xmp template ta(4,4)
!$xmp distribute ta(block,block) onto pa
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)
!$xmp align c(i,j) with ta(i,j)

  b = -2
  a = -1
  c = 6
  err = 0

!$xmp loop (i,j) on ta(i,j)
  do j=1, 4
   do i=1, 4
     a(i,j) = i*(j-1)
   enddo
  enddo

  call xmp_transpose(b, a, 0)
  call xmp_transpose(c, b, 0)

!$xmp loop (i,j) on ta(i,j) reduction(+:err)
  do j=1, 4
   do i=1, 4
      if(a(i,j) .ne. c(i,j)) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine

subroutine test_transpose_106()

  integer,parameter :: n=32
  integer a(n,n), b(n,n), c(n,n), rc(n,n)
  integer err
!$xmp nodes p(16)
!$xmp nodes pa(4,4)=p(1:16)
!$xmp template ta(n,n)
!$xmp distribute ta(cyclic,cyclic) onto pa
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)
!$xmp align c(i,j) with ta(i,j)

  b = -2
  a = -1
  c = 6
  err = 0

!$xmp loop (i,j) on ta(i,j)
  do j=1, n
   do i=1, n
     a(i,j) = i*(j-1)
   enddo
  enddo

  call xmp_transpose(b, a, 0)
  call xmp_transpose(c, b, 0)

!$xmp loop (i,j) on ta(i,j) reduction(+:err)
  do j=1, n
   do i=1, n
      if(a(i,j) .ne. c(i,j)) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine

!subroutine test_transpose_107()
!
!  integer,parameter :: n=32
!  integer a(n*2,n), b(n,n*2), c(n*2,n)
!  integer err
!!$xmp nodes p(16)
!!$xmp nodes pa(4,4)=p(1:16)
!!$xmp template ta(n*2,n)
!!$xmp template tb(n,n*2)
!!$xmp distribute ta(gblock((/32,16,8,8/)),block) onto pa
!!$xmp distribute tb(block,cyclic) onto pa
!!$xmp align a(i,j) with ta(i,j)
!!$xmp align b(i,j) with tb(i,j)
!!$xmp align c(i,j) with ta(i,j)
!
!  b = -2
!  a = -1
!  c = 6
!  err = 0
!
!!$xmp loop (i,j) on ta(i,j)
!  do j=1, n
!   do i=1, n*2
!     a(i,j) = i*(j-1)
!   enddo
!  enddo
!
!  call xmp_transpose(b, a, 0)
!  call xmp_transpose(c, b, 0)
!
!!$xmp loop (i,j) on ta(i,j) reduction(+:err)
!  do j=1, n
!   do i=1, n*2
!      if(a(i,j) .ne. c(i,j)) then
!         err = err+1
!      endif
!   enddo
!  enddo
!
!  call chk_int(err)
!
!end subroutine

subroutine test_transpose_108()

  integer a(4,4), b(4,4), c(4,4)
  integer err
!$xmp nodes p(4,4)
!$xmp template ta(4,4)
!$xmp distribute ta(block,block) onto p
!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with ta(i,j)
!$xmp align c(i,j) with ta(i,j)

  b = -2
  a = -1
  c = 1
  err = 0

!$xmp loop (i,j) on ta(i,j)
  do j=1, 4
   do i=1, 4
     a(i,j) = i*(j-1)
   enddo
  enddo

  call xmp_transpose(b, a, 0)
  call xmp_transpose(c, b, 0)

!$xmp loop (i,j) on ta(i,j) reduction(+:err)
  do j=1, 4
   do i=1, 4
      if(a(i,j) .ne. c(i,j)) then
         err = err+1
      endif
   enddo
  enddo

  call chk_int(err)

end subroutine
