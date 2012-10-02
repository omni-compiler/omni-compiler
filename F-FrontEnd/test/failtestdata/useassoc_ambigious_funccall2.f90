      module m1
      contains
        integer function f(a)
          integer :: a
        end function f
      end module m1

      module m2
      contains
        integer function f(a)
          integer :: a
        end function f
      end module m2

      program main
        use m1
        use m2
        integer :: i
        i = f(1)
      end program main

