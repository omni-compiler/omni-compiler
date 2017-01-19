      PROGRAM main
        PROCEDURE(REAL(KIND=8)), POINTER :: p

        INTERFACE
          FUNCTION f(a)
            REAL(KIND=8) :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE

        REAL(KIND=8) :: r

        p => f
        r = p(1)
      END PROGRAM main

      FUNCTION f(a)
        REAL(KIND=8) :: f
        INTEGER :: a
        f = a
      END FUNCTION f

