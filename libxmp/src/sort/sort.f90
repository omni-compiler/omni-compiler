program sort_block

  integer me, nprocs

  integer m(128)
  integer n = 0

  nprocs = xmp_num_nodes()

  do i = 1, nprocs
    m(i) = i * nprocs * 100;
    n = n + m(i);
  end do

  call sort_test(m, n, nprocs)

end program sort_block


subroutine sort_test(m, n, nprocs)

  integer me, nprocs

  integer m(nprocs)
  integer n

  real x
  integer r

!$xmp nodes p(*)

!$xmp template t0(n)
!$xmp distribute t0(gblock(m)) onto p

!$xmp template t1(n)
!$xmp distribute t1(block) onto p

  integer a0(n), b0(n)
!$xmp align a0(i) with t0(i)
!$xmp align b0(i) with t1(i)
!$xmp shadow b0(1:1)

  me = xmp_node_num() - 1

  !
  ! integer
  !

!$xmp loop on t0(i)
  do i = 1, n
     call random_number(x)
     a0(i) = mod(int(x * 1234), n)
  end do

  call xmp_sort_up(xmp_desc_of(a0), xmp_desc_of(b0))

!$xmp reflect (b0)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
!     write(*,*) i, b0(i)
     if (b0(i-1) > b0(i)) then
        r = 1
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

  if (me == 0) write(*,*) "PASS"

end program sort_test
