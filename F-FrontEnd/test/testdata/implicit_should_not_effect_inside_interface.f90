      SUBROUTINE SUBR (foo)
        IMPLICIT COMPLEX (k)
        INTERFACE
           FUNCTION foo ( a , k )
             REAL :: a ( 1 : k )
             ! k is integer
           END FUNCTION foo
        END INTERFACE
      END SUBROUTINE SUBR
