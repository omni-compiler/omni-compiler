      function func (a, b, c)
        real a, b, c, func
        func = a + b + c
      end function

      subroutine sub(x, y)
        real x, y
      end subroutine

      program main
        external func
        integer res
        res = func(1., c=3., b=2.)
        call sub(y=5., x=4.)
      end program
