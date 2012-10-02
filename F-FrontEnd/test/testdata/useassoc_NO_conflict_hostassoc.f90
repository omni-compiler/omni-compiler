      module m1
        integer :: i
      end module m1

      program main
        implicit none
        complex :: i
      contains
        subroutine sub()
          use m1
          i = i + 1
        end subroutine sub
      end program main
