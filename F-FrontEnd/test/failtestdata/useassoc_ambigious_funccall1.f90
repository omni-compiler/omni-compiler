      module m1
      contains
        integer function f()
        end function f
      end module m1

      module m2
      contains
        integer function f()
        end function f
      end module m2

      program main
        use m1
        use m2
        integer :: i
        i = f()
      end program main

