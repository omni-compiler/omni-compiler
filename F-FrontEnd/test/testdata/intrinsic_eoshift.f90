      real a(10), b
      real c(10,10)
      real d(10)
      integer e(10)
      a = eoshift(a, 1)             ! (ARRAY, SHIFT)
      c = eoshift(c, e)             ! (ARRAY, SHIFT)
      a = eoshift(a, 2, boundary=b) ! (ARRAY, SHIFT, BOUNDARY)
      c = eoshift(c, 2, d)          ! (ARRAY, SHIFT, BOUNDARY)
      c = eoshift(c, e, d)          ! (ARRAY, SHIFT, BOUNDARY)
      c = eoshift(c, 1, b, 2)       ! (ARRAY, SHIFT, BOUNDARY, DIM)
      c = eoshift(c, e, b, 2)       ! (ARRAY, SHIFT, BOUNDARY, DIM)
      end
