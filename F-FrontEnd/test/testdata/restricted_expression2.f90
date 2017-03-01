      subroutine foo(a,n)
          interface
              pure integer function id(a)
                  integer a
                  intent(in) :: a
              end function id
          end interface
          ! pure function call is restricted expression
          BLOCK
            character(len=id(8)) :: a
          END BLOCK
      end subroutine foo
