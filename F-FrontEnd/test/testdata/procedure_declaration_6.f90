       CALL XXX(PSS)
       CONTAINS
         SUBROUTINE XXX(PSI)
           PROCEDURE (INTEGER) :: PSI
           REAL  Y1
           Y1 = PSI()
           if(Y1.eq.10) THEN
              PRINT *, 'PASS'
           else
              PRINT *, 'NG'
              CALL EXIT(1)
           end if
         END SUBROUTINE XXX
         FUNCTION PSS()
           INTEGER :: PSS
           PSS = 10
         END FUNCTION PSS
       END
