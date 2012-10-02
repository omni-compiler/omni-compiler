      module m1
        type t
          integer :: n
        end type t
      end module m1

      module m2
        type t
          integer :: n
        end type t
      end module m2

      program main
        use m1; use m2
        type(t) :: a
        a%n = 1
      end program main

