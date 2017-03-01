       PROGRAM MAIN
         TYPE t(k)
           INTEGER, KIND :: k
           INTEGER(KIND=k) :: v
         END TYPE t
         TYPE(t(selected_int_kind(4 + 4))) :: v
       END PROGRAM MAIN
