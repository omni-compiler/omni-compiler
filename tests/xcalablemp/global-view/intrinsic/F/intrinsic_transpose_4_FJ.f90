program test_transpose_FJ

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
  call test_transpose_021()
  call test_transpose_022()
!  call test_transpose_023()

end program

subroutine test_transpose_004()

  integer a(16,16), b(16,16)
  integer error
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!!$xmp nodes p(4)
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
!!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!!$xmp nodes p(4)
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

subroutine test_transpose_021()

  integer,parameter:: M=23, N=31
  integer*4 a(M,N), b(N,M)
  integer error
!$xmp nodes p(4)
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
!$xmp nodes p(4)
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
!!$xmp nodes p(4)
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
