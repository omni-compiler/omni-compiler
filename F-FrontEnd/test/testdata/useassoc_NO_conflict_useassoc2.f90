      module m1
        integer :: i
      end module m1

      module m2
        use m1
        integer :: j
      end module m2

      module m3
        use m1
        integer :: k
      end module m3

      program main
        use m1
        use m2
        use m3
      end program main
