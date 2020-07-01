  integer function foo(n)
    integer n
    call bar(n)
    return

  contains
    subroutine bar(n)
      integer n
      integer foo

      foo = n
      return
    end subroutine bar

  end function foo

