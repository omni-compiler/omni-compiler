      module m1
        real*8,dimension(8,4,2,1) :: a
      end module m1

      module m2
        real*8,dimension(8,4,2,1) :: a
      end module m2

      program main
        use m1
        use m2
        a(1,1,1,1) = 1.0
      end program main

