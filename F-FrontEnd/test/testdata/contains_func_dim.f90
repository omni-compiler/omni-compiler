      program main
        integer::a

        !frontend is not supported foward reference
        !a = size(f())

        contains
          function f()
            integer,dimension(3)::f
            f(1) = 1
            f(2) = 2
            f(3) = 3
          end function
      end

