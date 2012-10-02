      module m1
        complex :: i
      end module m1

      program main
        implicit none
        use m1
      contains
        subroutine sub()
          integer :: i = 1
          i = i + 1
        end subroutine sub
      end program main
