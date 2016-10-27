      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
      CONTAINS
        FUNCTION f(this)
          IMPLICIT NONE
          CLASS(t) :: this
          TYPE(t) :: f
          f = t(this%v)
        END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t), TARGET :: a
        TYPE(t)  :: b
        CLASS(t), POINTER :: p
        a = t(1)
        p => a
        b = p%f()
      END PROGRAM main
