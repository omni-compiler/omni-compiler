      module b_11_53
          interface bFoo
              module procedure foo
          end interface
      contains
          subroutine foo()
              continue
          end subroutine foo
      end module b_11_53

      program main
          use b_11_53
          s=s+1
          call bFoo()
      end
