      INTERFACE
        FUNCTION f (i)
          CHARACTER(LEN=i+1) :: f
        END FUNCTION f
        FUNCTION g (i)
          CHARACTER(LEN=i+1) :: g
        END FUNCTION g
      END INTERFACE
      PROCEDURE(f), POINTER :: p
      p => g
      END
