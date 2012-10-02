      SUBROUTINE XXX( a, b )

!$OMP PARALLEL DO PRIVATE(I)
!$+ REDUCTION(MAX:K)
         DO 180 J = 1, 100


  180    CONTINUE
!$OMP END PARALLEL DO
      end subroutine XXX
