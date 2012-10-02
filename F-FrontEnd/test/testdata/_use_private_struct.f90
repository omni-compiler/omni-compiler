      MODULE use_private_struct
        IMPLICIT NONE

        PRIVATE :: tt, ttt, f
        PUBLIC :: u,p,v

        TYPE tt
           INTEGER*8 :: n
        END TYPE tt

        TYPE ttt
           TYPE(tt) :: p
           INTEGER :: n
        END TYPE ttt

        TYPE(tt)           :: v
        TYPE(tt),PARAMETER :: p = tt(1)
        TYPE(ttt)          :: u

      CONTAINS
        FUNCTION f()
          TYPE(tt) :: f
          f%n = 5
        END FUNCTION f
      END MODULE use_private_struct
