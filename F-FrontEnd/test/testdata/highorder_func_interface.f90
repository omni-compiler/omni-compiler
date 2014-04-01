      subroutine sub(p, n)
          integer n
          interface
              function p(m)
                  integer m
              end function p
          end interface
          i = p(3)
      end subroutine
