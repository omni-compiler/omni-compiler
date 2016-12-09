      PROGRAM main
        INTEGER :: asum = 10
        BLOCK
          INTEGER :: a(10)
          DO i=1,10
             a(i) = i
             asum = asum + a(i)
          END DO
          PRINT *, asum
        END BLOCK
        PRINT *, asum
      END PROGRAM main

