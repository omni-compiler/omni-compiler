      PROGRAM main
       CONTAINS
        SUBROUTINE sub()
          PROCEDURE(f), POINTER :: p
        END SUBROUTINE sub
        FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
        END FUNCTION f
      END PROGRAM main
