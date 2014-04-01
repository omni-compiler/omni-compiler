      module m
          real x(10)

      contains
          subroutine sub
              real z
              z = x(1)
          contains
              function x(n)
                  x = 1.23
                  return
              end function x
          end subroutine sub
      end module m
