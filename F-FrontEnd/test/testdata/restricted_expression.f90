      subroutine foo(c, d)
          character(len=1) :: c
          ! intrinsic function call is restricted expression
          ! constant is restricted expression
          ! dummy argument is restricted expression
          character(len=scan('123456789', c)) :: d
      end subroutine foo
