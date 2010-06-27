      module m
        implicit real(8) (F)
      contains
        subroutine s(g)
          real(8) :: g
        end subroutine s
        subroutine t()
          f = 1.0
          call s(f)
        end subroutine t
      end module m

