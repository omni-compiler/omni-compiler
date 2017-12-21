      INTERFACE
        FUNCTION f(a)
          INTEGER :: f, a
        END FUNCTION f
      END INTERFACE
      PROCEDURE(f), POINTER :: p
      END
