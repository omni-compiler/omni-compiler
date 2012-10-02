      module m
      contains
        subroutine sub()
        contains
          subroutine sub1()
            logical :: i
            i = f()
          end subroutine sub1
          logical function f()
            f = .TRUE.
          end function f
        end subroutine sub
      end module m
