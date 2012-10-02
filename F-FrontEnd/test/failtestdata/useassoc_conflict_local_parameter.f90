      module m1
        integer :: i
      end module m1

      program main
        use m1
        parameter (i = 1)
      end program main
