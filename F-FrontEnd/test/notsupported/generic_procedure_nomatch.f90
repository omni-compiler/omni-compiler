      module bFooModule
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
      end module bFooModule

      program main
          use bFooModule
          call bFoo()
      end
