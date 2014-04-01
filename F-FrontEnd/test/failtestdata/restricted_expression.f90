      subroutine foo(a,n)
          integer n
          ! internal function call is not restricted expression
          character(len=id(n)) :: a
      contains
          pure function id(a)
              integer id, a
              intent(in) :: a
          end function id
      end subroutine foo

