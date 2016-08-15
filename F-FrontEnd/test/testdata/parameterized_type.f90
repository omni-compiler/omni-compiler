
      PROGRAM MAIN
        TYPE st(dim, lg)
           INTEGER, KIND :: dim
           INTEGER, LEN :: lg
           REAL :: array(dim)
           CHARACTER(LEN=lg) :: ch
        END TYPE st

        TYPE(st(dim=2,len=4)) :: a

        TYPE(st):: b

        b = st(dim=2,len=5)(array=(/1,2/), ch="hoge")

      END PROGRAM MAIN
