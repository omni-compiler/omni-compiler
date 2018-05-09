      INTERFACE
        FUNCTION f() BIND(C)
        END FUNCTION f
      END INTERFACE
      PROCEDURE(f), BIND(C), POINTER :: p
      END
