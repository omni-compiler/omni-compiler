      module m
        implicit real(8) (g) 
        private
      contains
      function f() result(w)
        real(8) w
        w = 1.0
        w = sign(w, g())
      end function f
      end module m
