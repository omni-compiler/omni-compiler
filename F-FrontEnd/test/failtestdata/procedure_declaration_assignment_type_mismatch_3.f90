      PROGRAM main
        INTERFACE
          FUNCTION  f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
          SUBROUTINE sub()
          END SUBROUTINE sub
        END INTERFACE
        PROCEDURE(), POINTER :: i
        PROCEDURE(), POINTER :: j
        j => f
        j => sub
      END PROGRAM main
