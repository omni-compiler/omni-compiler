      PROGRAM test
        TYPE tt
           INTEGER :: a
           INTEGER :: b
           CHARACTER(LEN=20) s
           INTEGER c(10)
        END TYPE tt

        type(tt)::t
        t = tt(1, 2, "Happy", 3)
        t = tt(a=1, b=2, s="Happy", c=3)
        t = tt(s="Happy", b=2, c=3, a=1)
        t = tt(1, 2, c=3, s="Happy")

        print *, t
      end PROGRAM test
