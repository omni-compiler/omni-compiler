      integer function sumall(a, b, c,                                  & to be allowed
     &   d, e,                                                           ! to be allowed
     &   f, g                                                              @ warning expected
     &   ) result(val)
      integer a,b,c,d,e,f,g,val
      val = a+b+c+d+e+f+g
      end function

      if (sumall(1,2,3,4,5,6,7)/=28) then
         write(*,*) "NG",sumall(1,2,3,4,5,6,7),28
         stop 1
      else
         write(*,*) "OK"
      endif
      end



