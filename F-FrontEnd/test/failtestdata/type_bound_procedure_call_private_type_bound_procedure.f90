      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
           PROCEDURE, PRIVATE :: g => f
        END TYPE t
      CONTAINS
        FUNCTION f(this, w) RESULT(ans)
          IMPLICIT NONE
          CLASS(t) :: this
          INTEGER :: w
          INTEGER :: ans
          ans = this%v + w
        END FUNCTION f

        SUBROUTINE sub()
          TYPE(t), TARGET :: a
          CLASS(t), POINTER :: b
          INTEGER :: v
          a = t(5)
          b => a
          v = b%g(5)
          PRINT *, v
        END SUBROUTINE SUB
      END MODULE m

      PROGRAM main
        USE M
        TYPE(t), TARGET :: a
        CLASS(t), POINTER :: b
        INTEGER :: v
        a = t(10)
        b => a

        v = b%f(5)
        PRINT *, v

        CALL sub()

        v = b%g(5)
        PRINT *, v

      END PROGRAM main
