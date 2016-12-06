      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
        TYPE, EXTENDS(t) :: tt
         CONTAINS
           PROCEDURE, PUBLIC :: f => ff
        END TYPE tt
      CONTAINS
        FUNCTION f(this, i)
          INTEGER :: f
          CLASS(t) :: this
          INTEGER :: i
          PRINT *, "call F"
        END FUNCTION f
        FUNCTION ff(this, i)
          INTEGER :: ff
          CLASS(tt) :: this
          INTEGER :: i
          PRINT *, "call FF"
        END FUNCTION ff
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM main
