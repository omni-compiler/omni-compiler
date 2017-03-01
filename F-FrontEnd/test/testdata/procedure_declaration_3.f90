      PROGRAM main
        PROCEDURE(INTEGER), POINTER :: a
        PROCEDURE(f), POINTER :: b
        PROCEDURE(), POINTER :: i
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

        a => f
        PRINT *, a(1)
        a => g
        PRINT *, a(1, 2)
        b => f
        PRINT *, b(1)

        i => f
        PRINT *, i(1)
        i => g
        PRINT *, i(1,2)

      END PROGRAM main

      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
        f = 1 + 1
      END FUNCTION f

      FUNCTION g(a, b)
        INTEGER :: g
        INTEGER :: a
        INTEGER :: b
        g = 1 + a + b
      END FUNCTION g
