      module m
        implicit real(4) (g) 
        private
      contains
      function f() result(w)
        integer :: w
        w = maxval(g())
      end function f
      function g() result(w)
        integer, dimension(1:3) :: w
        w = 1
      end function g
      end module m
