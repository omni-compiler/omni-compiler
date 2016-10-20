      PROGRAM MAIN
        INTERFACE
           FUNCTION f(a)
             INTEGER :: f
             INTEGER :: a
           END FUNCTION f
        END INTERFACE
        BLOCK
          INTERFACE
             FUNCTION f(a)
               REAL :: f
               REAL :: a
             END FUNCTION f
          END INTERFACE
        END BLOCK
      END PROGRAM MAIN
