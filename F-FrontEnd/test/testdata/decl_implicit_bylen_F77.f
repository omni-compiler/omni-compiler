      program main
          implicit integer*8 (a, b)
          implicit real*8 (c, d)
          implicit complex*16 (e, f)
          dimension b(10)
          dimension d(10)
          dimension f(10)
          a = X'FFFFFFFFFFFFFFF'
          b(1) = X'FFFFFFFFFFFFFFF'
          c = dabs(c)
          d(1) = dabs(d(1))
          e = dabs(aimag(e))
          f(1) = dabs(aimag(f(1)))
      end
      
