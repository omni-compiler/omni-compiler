      module m
        implicit complex (a)
        integer :: a
      contains
        subroutine sub()
          real :: d
          d = CABS(a)
        end subroutine sub
      end module m
