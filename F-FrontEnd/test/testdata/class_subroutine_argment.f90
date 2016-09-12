      MODULE m
        TYPE t(k)
          INTEGER, KIND :: k
          INTEGER(KIND=k) :: v
        END TYPE t

        TYPE, EXTENDS(t) :: tt(l)
          INTEGER, LEN :: l
          CHARACTER(LEN=l) :: u
        END TYPE tt

      CONTAINS
        SUBROUTINE sub(v)
          CLASS(t(k=4)) :: v
        END SUBROUTINE sub
      END MODULE m
        
      PROGRAM main
        USE m
        TYPE(tt(k=4, l=8)) :: v
        CALL sub(v)
      END PROGRAM main
