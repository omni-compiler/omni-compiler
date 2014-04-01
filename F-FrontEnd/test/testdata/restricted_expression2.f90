      subroutine foo(a,n)
          interface
              pure integer function id(a)
                  integer a
                  intent(in) :: a
              end function id
          end interface
          integer n
          ! pure function call is restricted expression
          character(len=id(n)) :: a
      end subroutine foo
