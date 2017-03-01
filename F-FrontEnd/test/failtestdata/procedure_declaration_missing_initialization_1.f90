      PROGRAM main
        INTERFACE
           FUNCTION f(a)
             INTEGER :: f
             INTEGER :: a
           END FUNCTION f
        END INTERFACE
        PROCEDURE(f),POINTER :: g => h
      END PROGRAM
