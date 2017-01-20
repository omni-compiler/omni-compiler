      PROGRAM main
        TYPE t
          INTEGER :: v
          PROCEDURE(f),PASS(f),POINTER :: p => g
        END TYPE t
        INTERFACE
          FUNCTION f(a)
            CLASS(t) :: a
            TYPE(t) :: f
          END FUNCTION f
        END INTERFACE
      CONTAINS
        FUNCTION g(a)
          CLASS(t) :: a
          TYPE(t) :: g
        END FUNCTION g
      END PROGRAM main
