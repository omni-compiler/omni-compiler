      module m1
        type,private :: t
          integer :: n
        end type t
        type(t) :: a
      end module m1

      module m2
        type,private :: t
          integer :: n
        end type t
        type(t) :: a
      end module m2

      program main
        use m1
        use m2
        a%n = 1
      end program main

