! internal function/subroutine's type should not be defined in parent
      subroutine sub_p()
        integer sub
        contains
          subroutine sub ()
          end subroutine
      end subroutine
