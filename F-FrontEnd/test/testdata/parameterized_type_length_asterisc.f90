      PROGRAM MAIN
        TYPE st(lg)
           INTEGER, LEN :: lg
           CHARACTER(LEN=lg), POINTER :: ch
        END TYPE st
      CONTAINS
        SUBROUTINE sub(a1, a2)
          TYPE(st(lg=*)) :: a1
          TYPE(st(*)) :: a2
        END SUBROUTINE sub
      END PROGRAM MAIN
