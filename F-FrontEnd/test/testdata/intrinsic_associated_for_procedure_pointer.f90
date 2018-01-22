      INTERFACE
        FUNCTION f()
          INTEGER :: f
        END FUNCTION f
      END INTERFACE
      PROCEDURE(f), POINTER :: p
      p => f
      PRINT *, ASSOCIATED(p, f)
      END
