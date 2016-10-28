      PROGRAM MAIN
        TYPE t(k)
          INTEGER, KIND :: k
          INTEGER(KIND=k) :: v
        END TYPE t
        TYPE(t(k=selected_int_kind(5))) :: a
        TYPE(t(k=selected_int_kind(5))) :: b
        a = b
      END PROGRAM MAIN
