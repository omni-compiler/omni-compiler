      MODULE m
        TYPE t
          INTEGER :: v
          PROCEDURE(f),PASS(a),POINTER :: p 
        END TYPE t
       CONTAINS
        FUNCTION f(a, b)
          INTEGER :: f
          INTEGER :: a
          CLASS(t) :: b
        END FUNCTION f
      END MODULE m
      PROGRAM main
       
      END PROGRAM
