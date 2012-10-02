      module m1
        integer :: i
      end module m1

      program main
        use m1
        external :: i
      end program main
