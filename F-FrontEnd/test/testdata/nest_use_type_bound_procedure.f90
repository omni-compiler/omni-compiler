      MODULE mmm
        USE mm
       CONTAINS
        SUBROUTINE sss(a)
          TYPE(tt), POINTER :: a
          PRINT *, a%p%v
        END SUBROUTINE sss
      END MODULE mmm

