
      PROGRAM MAIN
        TYPE st(k, lg)
           INTEGER, KIND :: k
           INTEGER, LEN :: lg
           REAL(k) :: r
           CHARACTER(LEN=lg) :: ch
        END TYPE st

        TYPE(st(k=2,lg=4)) :: a

        TYPE(st(k=2,lg=5)):: b

        b = st(k=2,lg=5)(r=1.0, ch="hoge")

      END PROGRAM MAIN
