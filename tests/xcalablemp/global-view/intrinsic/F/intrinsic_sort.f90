module keys

  integer me, nprocs

  integer, parameter :: n = 1024
  integer, allocatable :: m(:)

!$xmp nodes p(*)

!$xmp template t0(n)
!$xmp distribute t0(block) onto p

  integer a0(n)
!$xmp align a0(i) with t0(i)

  real b0(n)
!$xmp align b0(i) with t0(i)

  real(8) c0(n)
!$xmp align c0(i) with t0(i)

end module keys


program sort_test

  use keys

  integer k, p

  me = xmp_node_num() - 1
  nprocs = xmp_num_nodes()

  allocate (m(nprocs))

  p = 0

  do i = 1, nprocs - 1
    m(i) = nprocs * 2 * i
    p = p + m(i)
  end do

  m(nprocs) = n - p

  call int_block
  call real_gblock
  call real8_cyclic

  if (me == 0) write(*,*) "PASS"

end program sort_test


subroutine int_block

  use keys

!$xmp template t1(n)
!$xmp distribute t1(block) onto p

  integer a1(n)
!$xmp align a1(i) with t0(i)
!$xmp shadow a1(1:1)

  real x
  integer r

!$xmp loop on t0(i)
  do i = 1, n
     call random_number(x)
     a0(i) = mod(int(x * 1234), n)
  end do

  call xmp_sort_up(xmp_desc_of(a0), xmp_desc_of(a1))

!$xmp reflect (a1)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
     if (a1(i-1) > a1(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

  call xmp_sort_down(xmp_desc_of(a0), xmp_desc_of(a1))

!$xmp reflect (a1)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
     if (a1(i-1) < a1(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

end subroutine int_cyclic


subroutine real_gblock

  use keys

!$xmp template t1(n)
!$xmp distribute t1(gblock(m)) onto p

  real b1(n)
!$xmp align b1(i) with t1(i)
!$xmp shadow b1(1:1)

  integer r

!$xmp loop on t0(i)
  do i = 1, n
     call random_number(b0(i))
  end do

  call xmp_sort_up(xmp_desc_of(b0), xmp_desc_of(b1))

!$xmp reflect (b1)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
     if (b1(i-1) > b1(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

  call xmp_sort_down(xmp_desc_of(b0), xmp_desc_of(b1))

!$xmp reflect (b1)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
     if (b1(i-1) < b1(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

end subroutine real_gblock


subroutine real8_cyclic

  use keys

!$xmp template t1(n)
!$xmp distribute t1(cyclic(4)) onto p

  real(8) c1(n)
!$xmp align c1(i) with t1(i)

  real(8) c2(n)
!$xmp align c2(i) with t0(i)
!$xmp shadow c2(1:1)



  integer r

!$xmp loop on t0(i)
  do i = 1, n
     call random_number(c0(i))
  end do

  call xmp_sort_up(xmp_desc_of(c0), xmp_desc_of(c1))

!$xmp gmove
  c2(:) = c1(:)

!$xmp reflect (c2)

  r = 0

!$xmp loop on t0(i) reduction(+:r)
  do i = 2, n
     if (c2(i-1) > c2(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

  call xmp_sort_down(xmp_desc_of(c0), xmp_desc_of(c1))

  r = 0

!$xmp gmove
  c2(:) = c1(:)

!$xmp reflect (c2)

!$xmp loop on t0(i) reduction(+:r)
  do i = 2, n
     if (c2(i-1) < c2(i)) then
        r = 1
        exit
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

end subroutine real8_cyclic
