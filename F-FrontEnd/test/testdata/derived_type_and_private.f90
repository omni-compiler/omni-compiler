      MODULE m
        TYPE t
           INTEGER, PRIVATE :: a
           INTEGER :: b
        END TYPE t
        TYPE(t) :: v0 = t(a=1, b=2)
        TYPE(t) :: v1 = t(1, 2)
      END MODULE m

      PROGRAM MAIN
      END PROGRAM MAIN
