      PROGRAM main
        INTERFACE
          SUBROUTINE sub()
          END SUBROUTINE sub
        END INTERFACE
        PROCEDURE(sub), POINTER :: p
        p => s
      END PROGRAM
