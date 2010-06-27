      subroutine sub() 
          integer p
          parameter(p = 10)
          integer a(p), b
          dimension b(p)
      end

      program main
          integer p, k
          parameter(p = 10, k = 4)
          integer a(p), b
          integer(k) n
          dimension b(p)
          character(len=p) c
      end
