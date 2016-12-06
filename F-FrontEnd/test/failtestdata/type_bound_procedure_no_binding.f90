      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
      CONTAINS
        FUNCTION g(this, w) RESULT(ans)
          IMPLICIT NONE
          CLASS(t) :: this
          INTEGER :: w
          INTEGER :: ans
          ans = this%v + w
        END FUNCTION g
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM main
