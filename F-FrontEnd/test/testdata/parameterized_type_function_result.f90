      PROGRAM MAIN
        TYPE :: t(k,l)
          INTEGER, KIND :: k
          INTEGER, LEN :: l
          INTEGER(KIND=k) :: a
          CHARACTER(LEN=l) :: c
        END TYPE t
      CONTAINS
        TYPE(t(4,8)) FUNCTION f(a)
           INTEGER :: a
        END FUNCTION f

      END PROGRAM
