      module m1
        integer :: i
      end module m1

      program main
        use m1
        integer :: a,b,c
        namelist /i/ a, b, c
      end program main
