      PROGRAM main
        INTEGER :: i
        REAL :: X(100), Y(100)

        DO CONCURRENT (i = 1:100)
           Y(i) =X(i)
        END DO
        DO , CONCURRENT (i = 1:100)
           Y(i) =X(i)
        END DO
        DO 1000 CONCURRENT (i = 1:100)
           Y(i) =X(i)
 1000   END DO
        DO 1001, CONCURRENT (i = 1:100)
           Y(i) =X(i)
 1001   END DO
        const1: DO CONCURRENT (i = 1:100)
           Y(i) =X(i)
        END DO const1
        const2: DO , CONCURRENT (i = 1:100)
           Y(i) =X(i)
        END DO const2
        const3: DO 1002 CONCURRENT (i = 1:100)
           Y(i) =X(i)
 1002   END DO const3
        const4: DO 1003 , CONCURRENT (i = 1:100)
           Y(i) =X(i)
 1003   END DO const4
      END PROGRAM main
