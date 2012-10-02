      module m1
        integer,dimension(3) :: i
      end module m1

      program main
        use m1
        allocatable :: i
      end program main
