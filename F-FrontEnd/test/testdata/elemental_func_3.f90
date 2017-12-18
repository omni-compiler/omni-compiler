      MODULE m
        TYPE t
          INTEGER :: v = 0
         CONTAINS
          PROCEDURE, PASS :: f
        END TYPE t
       CONTAINS
        ELEMENTAL FUNCTION f(this)
          CLASS(t), INTENT(IN) :: this
          INTEGER :: f
          f = this%v
        END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t), DIMENSION(1:3) :: a
        a(1)%v = 1; a(2)%v = 2; a(3)%v = 3; 
        PRINT *, a%f() !=> 1 2 3
      END PROGRAM main
