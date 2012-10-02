      program main

      integer, allocatable :: a(:)
      integer :: st(10)
      type t0
          integer :: t_st(10)
      end type t0
      type(t0) :: t(10)

      allocate(a(10), stat = t(1)%t_st(1))

      end program main

