      function f()
        integer,dimension(3)::f
        f(1) = 1
        f(2) = 2
        f(3) = 3
      end function

      program main
        integer::a
        interface
          function f()
            integer,dimension(3)::f
          end function
        end interface

        a = size(f())
      end

