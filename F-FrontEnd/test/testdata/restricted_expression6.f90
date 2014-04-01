      subroutine foo(a,n)
          integer n, m
          common /dat/ m
          ! object designator is in common block is restricted expression
          character(len=n+m) :: a
      end subroutine foo
