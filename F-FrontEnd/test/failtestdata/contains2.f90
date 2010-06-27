! same function name as sibling is not permitted
      function func1()
        func1 = 1.
        contains
          subroutine func ()
          end subroutine
          subroutine func ()
          end subroutine
      end function

