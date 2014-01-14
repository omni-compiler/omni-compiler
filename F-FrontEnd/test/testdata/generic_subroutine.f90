      module generic_subroutine
          interface bFoo
              module procedure foo, bar
          end interface
      contains
          subroutine foo(a)
              integer a
              continue
          end subroutine foo
          subroutine bar(b)
              real b
              continue
          end subroutine bar
      end module generic_subroutine

      program main
          use generic_subroutine
          real s
          call bFoo(s)
      end
