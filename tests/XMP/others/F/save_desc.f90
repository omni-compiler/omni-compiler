program main

!$xmp nodes p(4)

  double precision t0, t1, MPI_Wtime

  t0 = MPI_Wtime()
  do i = 1, 10000
     call sub1
  end do
  t1 = MPI_Wtime()
!$xmp task on p(1) nocomm
  write(*,*) "no save_desc:", t1 - t0
!$xmp end task

  t0 = MPI_Wtime()
  do i = 1, 10000
     call sub2
  end do
  t1 = MPI_Wtime()
!$xmp task on p(1) nocomm
  write(*,*) "save_desc:", t1 - t0
!$xmp end task

!$xmp task on p(1) nocomm
  write(*,*) "PASS"
!$xmp end task

end program main

subroutine sub1

!$xmp nodes p(4)
!$xmp template t(100)
!$xmp distribute t(block) onto p
  real a(100), b(100), c(100)
!$xmp align (i) with t(i) :: a, b, c

!$xmp array on t
  a = 0.
!$xmp array on t
  b = 0.
!$xmp array on t
  c = 0.

!$xmp loop on t(i)
  do i = 1, 100
     a(i) = a(i) + b(i) * c(i)
  end do

end subroutine sub1

subroutine sub2

!$xmp nodes p(4)
!$xmp save_desc p

!$xmp template t(100)
!$xmp save_desc t
!$xmp distribute t(block) onto p

  real a(100), b(100), c(100)
!$xmp align (i) with t(i) :: a, b, c
!$xmp save_desc :: a, b, c

!$xmp array on t
  a = 0.
!$xmp array on t
  b = 0.
!$xmp array on t
  c = 0.

!$xmp loop on t(i)
  do i = 1, 100
     a(i) = a(i) + b(i) * c(i)
  end do

end subroutine sub2
