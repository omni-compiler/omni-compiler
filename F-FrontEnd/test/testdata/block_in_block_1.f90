      PROGRAM MAIN
        IMPLICIT NONE
        REAL :: a = 1.0
        BLOCK
          INTEGER :: a = 1
          BLOCK
            INTEGER :: j
            j = IABS(a)
          END BLOCK
        END BLOCK
      END PROGRAM MAIN
