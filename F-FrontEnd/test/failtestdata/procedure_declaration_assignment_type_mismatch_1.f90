      PROGRAM main
        PROCEDURE(REAL(KIND=8)), POINTER :: p

        INTERFACE
          FUNCTION f(a)
            REAL(KIND=16) :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE

        p => f
      END PROGRAM main

      FUNCTION f(a)
        REAL(KIND=16) :: f
        INTEGER :: a
        f = a
      END FUNCTION f

