      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
           PROCEDURE, PUBLIC :: g => f, h => f
        END TYPE t
      CONTAINS
        FUNCTION f(this, w) RESULT(ans)
          IMPLICIT NONE
          CLASS(t) :: this
          INTEGER :: w
          INTEGER :: ans
          ans = this%v + w
        END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
        INTEGER :: a
        TYPE(t) :: v = t(1)
        a = v%f(1)
      END PROGRAM main
