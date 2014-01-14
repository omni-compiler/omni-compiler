      real a(10)
      real b(10,10)
      integer c(10)
      a = cshift(a, 1)             ! (ARRAY, SHIFT)
      b = cshift(b, c)             ! (ARRAY, SHIFT)
      b = cshift(b, 1, 2)          ! (ARRAY, SHIFT, DIM)
      b = cshift(b, c, 2)          ! (ARRAY, SHIFT, DIM)
      end
