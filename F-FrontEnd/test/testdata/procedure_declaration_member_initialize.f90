      PROGRAM main
        TYPE t
          INTEGER :: v
          PROCEDURE(p),PASS(a),POINTER :: p => p
        END TYPE t
      CONTAINS
        FUNCTION p(a)
          CLASS(t) :: a
          TYPE(t) :: p
        END FUNCTION p
      END PROGRAM main
