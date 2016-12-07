      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
           PROCEDURE, PUBLIC :: g
           GENERIC :: p => f, g
        END TYPE t
      CONTAINS
        FUNCTION f(this, i)
          INTEGER :: f
          CLASS(t) :: this
          INTEGER :: i
          PRINT *, "call F"
          f = this%g(1.0)
        END FUNCTION f
        FUNCTION g(this, r)
          CLASS(t) :: this
          REAL :: r
          REAL :: g
          PRINT *, "call G"
          g = 1.0
        END FUNCTION g
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM main
