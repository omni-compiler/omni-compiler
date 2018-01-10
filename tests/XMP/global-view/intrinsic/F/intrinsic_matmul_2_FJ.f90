program test_matmul_FJ

  call test_matmul_004()
!  call test_matmul_005()
  call test_matmul_006()
!  call test_matmul_007()

end program

subroutine test_matmul_004()

!$xmp nodes p(2)
!$xmp template ta(16)
!$xmp template tb(8)
!$xmp template tx(16)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(block) onto p
!$xmp distribute tx(block) onto p

  integer*4 a(16,8), b(8,16), x(16,16)
  integer*4 c(16,8), d(8,16), y(16,16)
  integer*4 error

!$xmp align a(i,*) with ta(i)
!$xmp align b(i,*) with tb(i)
!$xmp align x(i,*) with tx(i)

!$xmp loop (i) on ta(i)
  do j=1, 8
     do i=1, 16
        a(i,j) = (j-1)*16+i-1
     enddo
  enddo
  do j=1, 8
     do i=1, 16
        c(i,j) = (j-1)*16+i-1
     enddo
  enddo

!$xmp loop (i) on tb(i)
  do j=1, 16
     do i=1, 8
        b(i,j) = -1*((j-1)*8+i-1)
     enddo
  enddo
  do j=1, 16
     do i=1, 8
        d(i,j) = -1*((j-1)*8+i-1)
     enddo
  enddo

  y = matmul(c, d)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (i) on tx(i) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

!subroutine test_matmul_005()
!
!!$xmp nodes p(2)
!!$xmp template ta(16)
!!$xmp template tb(8)
!!$xmp template tx(16)
!!$xmp distribute ta(gblock((/2,14/))) onto p
!!$xmp distribute tb(gblock((/6,2/))) onto p
!!$xmp distribute tx(gblock((/2,14/))) onto p
!
!  integer*4 a(16,8), b(8,16), x(16,16)
!  integer*4 c(16,8), d(8,16), y(16,16)
!  integer*4 error
!
!!$xmp align a(i,*) with ta(i)
!!$xmp align b(i,*) with tb(i)
!!$xmp align x(i,*) with tx(i)
!
!!$xmp loop (i) on ta(i)
!  do j=1, 8
!     do i=1, 16
!        a(i,j) = (j-1)*16+i-1
!     enddo
!  enddo
!  do j=1, 8
!     do i=1, 16
!        c(i,j) = (j-1)*16+i-1
!     enddo
!  enddo
!
!!$xmp loop (i) on tb(i)
!  do j=1, 16
!     do i=1, 8
!        b(i,j) = -1*((j-1)*8+i-1)
!     enddo
!  enddo
!  do j=1, 16
!     do i=1, 8
!        d(i,j) = -1*((j-1)*8+i-1)
!     enddo
!  enddo
!
!  y = matmul(c, d)
!  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))
!
!  error = 0
!!$xmp loop (i) on tx(i) reduction(+: error)
!  do j=1, 16
!     do i=1, 16
!        if(x(i,j) .ne. y(i,j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int(error)
!
!end subroutine

subroutine test_matmul_006()

!$xmp nodes p(2)
!$xmp template ta(8)
!$xmp template tb(16)
!$xmp template tx(16)
!$xmp distribute ta(block) onto p
!$xmp distribute tb(block) onto p
!$xmp distribute tx(block) onto p

  integer*4 a(16,8), b(8,16), x(16,16)
  integer*4 c(16,8), d(8,16), y(16,16)
  integer*4 error

!$xmp align a(*,i) with ta(i)
!$xmp align b(*,i) with tb(i)
!$xmp align x(*,i) with tx(i)

!$xmp loop (j) on ta(j)
  do j=1, 8
     do i=1, 16
        a(i,j) = (j-1)*16+i-1
     enddo
  enddo
  do j=1, 8
     do i=1, 16
        c(i,j) = (j-1)*16+i-1
     enddo
  enddo

!$xmp loop (j) on tb(j)
  do j=1, 16
     do i=1, 8
        b(i,j) = -1*((j-1)*8+i-1)
     enddo
  enddo
  do j=1, 16
     do i=1, 8
        d(i,j) = -1*((j-1)*8+i-1)
     enddo
  enddo

  y = matmul(c, d)
  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))

  error = 0
!$xmp loop (j) on tx(j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

!subroutine test_matmul_007()
!
!!$xmp nodes p(2)
!!$xmp template ta(8)
!!$xmp template tb(16)
!!$xmp template tx(16)
!!$xmp distribute ta(gblock((/3,5/))) onto p
!!$xmp distribute tb(gblock((/2,14/))) onto p
!!$xmp distribute tx(gblock((/2,14/))) onto p
!
!  integer*8 a(16,8), b(8,16), x(16,16)
!  integer*8 c(16,8), d(8,16), y(16,16)
!  integer*4 error
!
!!$xmp align a(*,i) with ta(i)
!!$xmp align b(*,i) with tb(i)
!!$xmp align x(*,i) with tx(i)
!
!!$xmp loop (j) on ta(j)
!  do j=1, 8
!     do i=1, 16
!        a(i,j) = (j-1)*16+i-1
!     enddo
!  enddo
!  do j=1, 8
!     do i=1, 16
!        c(i,j) = (j-1)*16+i-1
!     enddo
!  enddo
!
!!$xmp loop (j) on tb(j)
!  do j=1, 16
!     do i=1, 8
!        b(i,j) = -1*((j-1)*8+i-1)
!     enddo
!  enddo
!  do j=1, 16
!     do i=1, 8
!        d(i,j) = -1*((j-1)*8+i-1)
!     enddo
!  enddo
!
!  y = matmul(c, d)
!  call xmp_matmul(xmp_desc_of(x), xmp_desc_of(a), xmp_desc_of(b))
!
!  error = 0
!!$xmp loop (j) on tx(j) reduction(+: error)
!  do j=1, 16
!     do i=1, 16
!        if(x(i,j) .ne. y(i,j)) error = error+1
!     enddo
!  enddo
!
!  call chk_int(error)
!
!end subroutine
