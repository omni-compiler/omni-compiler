      program test_allocated
        integer :: i = 4
        real(4), allocatable :: x(:)
        if (allocated(x) .eqv. .false.) allocate(x(i))
      end program test_allocated
