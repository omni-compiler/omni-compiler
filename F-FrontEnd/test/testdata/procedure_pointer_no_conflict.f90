      PROGRAM main
        INTERFACE
          FUNCTION f()
          END FUNCTION
        END INTERFACE
        PROCEDURE(f), POINTER :: p
       CONTAINS
        SUBROUTINE sub()
        INTERFACE
          FUNCTION f()
          END FUNCTION
        END INTERFACE
        PROCEDURE(f),POINTER :: p
        END SUBROUTINE
      END
