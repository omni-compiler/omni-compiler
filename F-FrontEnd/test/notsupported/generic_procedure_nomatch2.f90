      module bFooModule
          interface bFoo
              module procedure foo
          end interface
      contains
          subroutine foo(a)
              real a
              continue
          end subroutine foo
      end module bFooModule

      program main
          use bFooModule
          integer a
          call bFoo(a)
      end
