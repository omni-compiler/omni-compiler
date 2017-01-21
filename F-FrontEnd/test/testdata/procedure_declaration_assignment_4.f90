      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
        f = a
      END FUNCTION f

      SUBROUTINE sub()
      END SUBROUTINE sub

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
        i => f
        j => sub
      END PROGRAM main
