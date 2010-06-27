      program main
        real r
        integer i
        interface fun
          real function rfun (x)
            real, intent(in) :: x
          end function rfun
          real function ifun (x)
            integer, intent(in) :: x
          end function ifun
        end interface
        r = fun(2.0)
        i = fun(1)
      end program main

      real function rfun (x)
        real, intent(in) :: x
        rfun = 3.0 * x
      end function rfun

      integer function ifun (x)
        integer, intent(in) :: x
        ifun = 3 * x
      end function ifun
