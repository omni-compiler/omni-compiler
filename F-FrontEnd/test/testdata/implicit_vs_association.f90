      module m
        implicit complex (a)
        integer :: a
      contains
        subroutine sub(a)
          real :: d
          d = CABS(a)
        end subroutine sub
      end module m
