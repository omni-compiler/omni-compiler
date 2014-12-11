program test_matmul_FJ

  call test_matmul_001()
  call test_matmul_002()
  call test_matmul_003()
  call test_matmul_008()
  call test_matmul_009()

end program

subroutine test_matmul_001()

!$xmp nodes p(2,2)
!$xmp template ta(16,8)
!$xmp template tb(8,16)
!$xmp template tx(16,16)
!$xmp distribute ta(block,block) onto p
!$xmp distribute tb(cyclic,cyclic) onto p
!$xmp distribute tx(cyclic(2),cyclic(3)) onto p

  integer*4 a(16,8), b(8,16), x(16,16)
  integer*4 c(16,8), d(8,16), y(16,16)
  integer*4 error

!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j)
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

!$xmp loop (i,j) on tb(i,j)
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
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_002()

!$xmp nodes p(2,2)
!$xmp template ta(16,8)
!$xmp template tb(8,16)
!$xmp template tx(16,16)
!$xmp distribute ta(cyclic(3),cyclic(3)) onto p
!$xmp distribute tb(block,block) onto p
!$xmp distribute tx(cyclic,cyclic) onto p

  integer a(16,8), b(8,16), x(16,16)
  integer c(16,8), d(8,16), y(16,16)
  integer error

!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j)
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

!$xmp loop (i,j) on tb(i,j)
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
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_003()

!$xmp nodes p(2,2)
!$xmp template ta(16,8)
!$xmp template tb(8,16)
!$xmp template tx(16,16)
!$xmp distribute ta(cyclic,cyclic) onto p
!$xmp distribute tb(cyclic(3),cyclic(3)) onto p
!$xmp distribute tx(block,block) onto p

  integer a(16,8), b(8,16), x(16,16)
  integer c(16,8), d(8,16), y(16,16)
  integer error

!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j)
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

!$xmp loop (i,j) on tb(i,j)
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
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_008()

!$xmp nodes p(2,2)
!$xmp template ta(16,8)
!$xmp template tb(8,16)
!$xmp template tx(16,16)
!$xmp distribute ta(block,block) onto p
!$xmp distribute tb(block,block) onto p
!$xmp distribute tx(block,block) onto p

  integer*4 a(16,8), b(8,16), x(16,16)
  integer*4 c(16,8), d(8,16), y(16,16)
  integer*4 error

!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j)

!$xmp loop (i,j) on ta(i,j)
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

!$xmp loop (i,j) on tb(i,j)
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
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine

subroutine test_matmul_009()

!$xmp nodes p(2,2)
!$xmp template ta(16,8)
!$xmp template tb(8,16)
!$xmp template tx(16,16)
!$xmp distribute ta(block,block) onto p
!$xmp distribute tb(block,block) onto p
!$xmp distribute tx(block,block) onto p

  integer*4 a(16,8), b(8,16), x(16,16)
  integer*4 c(16,8), d(8,16), y(16,16)
  integer*4 error

!$xmp align a(i,j) with ta(i,j)
!$xmp align b(i,j) with tb(i,j)
!$xmp align x(i,j) with tx(i,j)
!$xmp shadow a(1,1)
!$xmp shadow b(1,1)
!$xmp shadow x(1,1)

!$xmp loop (i,j) on ta(i,j)
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

!$xmp loop (i,j) on tb(i,j)
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
!$xmp loop (i,j) on tx(i,j) reduction(+: error)
  do j=1, 16
     do i=1, 16
        if(x(i,j) .ne. y(i,j)) error = error+1
     enddo
  enddo

  call chk_int(error)

end subroutine
