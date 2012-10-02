      module m1
        integer :: i
      end module m1

      module m2
        real :: i
      end module m2

      program main
        use m1
        use m2
        i = 1
      end program main
