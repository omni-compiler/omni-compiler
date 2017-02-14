       PROGRAM main
        CALL sub
        CONTAINS
         SUBROUTINE sub()
           PROCEDURE(f), POINTER :: p
           p => f
           if(p(20).eq.20) then
             PRINT *, 'PASS'
           else
             PRINT *, 'NG'
             CALL EXIT(1)
           end if 
         END SUBROUTINE sub
         FUNCTION f(a)
           INTEGER :: f
           INTEGER :: a
           f = a
         END FUNCTION f
       END PROGRAM main
