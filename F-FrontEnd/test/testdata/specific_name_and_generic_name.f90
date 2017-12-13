      MODULE m1
        INTERFACE g
          MODULE PROCEDURE h
        END INTERFACE g
       CONTAINS
        FUNCTION h(a)
          COMPLEX :: h, a
        END FUNCTION h
        FUNCTION g(a)
          INTEGER :: g, a
        END FUNCTION g
      END MODULE m1

      !MODULE m2
      !  TYPE t
      !    INTEGER :: v
      !  END TYPE t
      !  INTERFACE t
      !    MODULE PROCEDURE f
      !  END INTERFACE t
      ! CONTAINS
      !  FUNCTION f(a)
      !    INTEGER :: f, a
      !  END FUNCTION f
      !END MODULE m2
