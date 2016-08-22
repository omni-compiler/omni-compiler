      MODULE PARAMETERIZED_DERIVED_TYPE
        TYPE :: t(k,l)
          INTEGER, KIND :: k = 8
          INTEGER, LEN :: l = 10
          INTEGER(KIND=k) :: a = 8
          CHARACTER(LEN=l) :: b = "a"
        END TYPE t
        TYPE :: tt
          TYPE(t) :: v = t(k=8, l=24)(8, "b")
        END TYPE tt
      END MODULE PARAMETERIZED_DERIVED_TYPE

      PROGRAM MAIN
      END PROGRAM
