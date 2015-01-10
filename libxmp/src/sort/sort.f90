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

  real a1(n), b1(n)
!$xmp align a1(i) with t0(i)
!$xmp align b1(i) with t1(i)
!$xmp shadow b1(1:1)

  real(8) a2(n), b2(n)
!$xmp align a2(i) with t0(i)
!$xmp align b2(i) with t1(i)
!$xmp shadow b2(1:1)

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
     if (b0(i-1) > b0(i)) then
        r = 1
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

  call xmp_sort_down(xmp_desc_of(a0), xmp_desc_of(b0))

!$xmp reflect (b0)

  r = 0

!$xmp loop on t1(i) reduction(+:r)
  do i = 2, n
     if (b0(i-1) < b0(i)) then
        r = 1
     end if
  end do

  if (r > 0) then
     if (me == 0) write(*,*) "ERROR"
     call exit(1)
  end if

! !
! ! real
! !

! !$xmp loop on t0(i)
!   do i = 1, n
!      call random_number(a1(i))
!   end do

!   call xmp_sort_up(xmp_desc_of(a1), xmp_desc_of(b1))

! !$xmp reflect (b1)

!   r = 0

! !$xmp loop on t1(i) reduction(+:r)
!   do i = 2, n
!      if (b1(i-1) > b1(i)) then
!         r = 1
!      end if
!   end do

!   if (r > 0) then
!      if (me == 0) write(*,*) "ERROR"
!      call exit(1)
!   end if

!   call xmp_sort_down(xmp_desc_of(a1), xmp_desc_of(b1))

! !$xmp reflect (b1)

!   r = 0

! !$xmp loop on t1(i) reduction(+:r)
!   do i = 2, n
!      if (b1(i-1) < b1(i)) then
!         r = 1
!      end if
!   end do

!   if (r > 0) then
!      if (me == 0) write(*,*) "ERROR"
!      call exit(1)
!   end if

! !
! ! real(8)
! !

! !$xmp loop on t0(i)
!   do i = 1, n
!      call random_number(a2(i))
!   end do

!   call xmp_sort_up(xmp_desc_of(a2), xmp_desc_of(b2))

! !$xmp reflect (b2)

!   r = 0

! !$xmp loop on t1(i) reduction(+:r)
!   do i = 2, n
!      if (b2(i-1) > b2(i)) then
!         r = 1
!      end if
!   end do

!   if (r > 0) then
!      if (me == 0) write(*,*) "ERROR"
!      call exit(1)
!   end if

!   call xmp_sort_down(xmp_desc_of(a2), xmp_desc_of(b2))

! !$xmp reflect (b2)

!   r = 0

! !$xmp loop on t1(i) reduction(+:r)
!   do i = 2, n
!      if (b2(i-1) < b2(i)) then
!         r = 1
!      end if
!   end do

!   if (r > 0) then
!      if (me == 0) write(*,*) "ERROR"
!      call exit(1)
!   end if

!
!
!

  if (me == 0) write(*,*) "PASS"

end program sort_test
