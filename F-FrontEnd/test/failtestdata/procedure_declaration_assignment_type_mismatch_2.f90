      PROGRAM main
        PROCEDURE(f), POINTER :: p

        INTERFACE
          FUNCTION f(a)
            REAL(KIND=16) :: f
            INTEGER :: a
          END FUNCTION f
          FUNCTION g(a)
            REAL(KIND=8) :: g
            INTEGER :: a
          END FUNCTION g
        END INTERFACE

        p => g
      END PROGRAM main

      FUNCTION f(a)
        REAL(KIND=16) :: f
        INTEGER :: a
        f = a
      END FUNCTION f

