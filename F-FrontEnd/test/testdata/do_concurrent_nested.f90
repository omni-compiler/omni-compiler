      PROGRAM main
        INTEGER, PARAMETER :: M = 10, N = 10
        INTEGER, DIMENSION(M,N) :: A, B, C
        DO CONCURRENT (I = 1:M)
          DO CONCURRENT (J = 1:N)
            C (I, J) = A (I, J) + B(I, J)
          END DO
        END DO
      END PROGRAM main
