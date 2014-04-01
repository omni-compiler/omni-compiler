      module mod
        contains
          function hoge(x)
            integer :: x
            hoge = fuge(x) ** 2 + fuge(x)
          end function
          function fuge(y)
            integer :: y
            fuge = y + 1
          end function
      end module
