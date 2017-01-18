      PROGRAM main
        PROCEDURE(f), POINTER :: a
        INTERFACE
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
          FUNCTION g(a, b)
            INTEGER :: g
            INTEGER :: a
            INTEGER :: b
          END FUNCTION g
        END INTERFACE

        a => g
        PRINT *, g(1, 2)
      END PROGRAM main
