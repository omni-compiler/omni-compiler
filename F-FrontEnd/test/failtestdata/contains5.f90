! internal function/subroutine should not contain 'contains'
      function func1()
        func1 = 1.
        contains
          subroutine sub1 ()
            contains
            subroutine sub2 ()
            end subroutine
          end subroutine
      end function

