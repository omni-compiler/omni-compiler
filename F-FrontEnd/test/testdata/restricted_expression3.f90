      subroutine foo(a,n)
          integer n
          ! array constructor is restricted expression
          ! specification inquiry is restricted expression
          character(len=size((/1, 2, 3/))) :: a
      end subroutine foo
