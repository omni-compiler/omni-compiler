      program main
        real f
        interface
          real function fun (x)
            real, intent(in) :: x
          end function fun
        end interface
        f = fun(2.0)
      end program

      real function fun (x)
        real, intent(in) :: x
        fun = 3.0 * x
      end function
