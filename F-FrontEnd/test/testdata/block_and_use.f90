      MODULE block_and_use
        REAL :: i
      END MODULE block_and_use

      PROGRAM MAIN
        IMPLICIT NONE
        BLOCK
          USE block_and_use
          INTEGER :: j
          i = 2.5
          j = EXPONENT(i)
        END BLOCK
      END PROGRAM MAIN

