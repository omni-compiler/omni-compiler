      subroutine foo(a,n)
          integer n
          ! implied do with restricted expression
          ! specification inquiry is restricted expression
          character(len=size((/ (i, i = 0, 2, 3) /))) :: a
      end subroutine foo
