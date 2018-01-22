      MODULE m
        IMPLICIT NONE
        TYPE t; INTEGER :: v; END TYPE t
       CONTAINS
        ELEMENTAL FUNCTION f(x) RESULT(r)
          IMPLICIT NONE
          REAL(KIND=4), INTENT(IN) :: x
          TYPE(t) :: r
        END FUNCTION f
      END MODULE m
