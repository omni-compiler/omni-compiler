      MODULE m
        TYPE :: t(k,l)
          INTEGER, KIND :: k
          INTEGER, LEN :: l
          INTEGER(KIND=k) :: a = 1
          CHARACTER(LEN=l) :: c = "aaa"
        END TYPE t
      END MODULE m

      PROGRAM MAIN
        USE m
        IMPLICIT TYPE(t(8,8)) (a)
        a = t(8,8)(1,"bb")
      END PROGRAM
