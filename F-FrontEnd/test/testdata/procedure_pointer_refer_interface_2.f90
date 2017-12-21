      INTERFACE
        FUNCTION f (p)
          INTERFACE
           FUNCTION g(a)
             INTEGER :: g, a
           END FUNCTION g
          END INTERFACE
          PROCEDURE(g), POINTER :: p
          INTEGER :: f
        END FUNCTION f
      END INTERFACE
      END


