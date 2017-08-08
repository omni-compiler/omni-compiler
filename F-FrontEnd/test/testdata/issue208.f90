MODULE mod1

IMPLICIT NONE

TYPE typ1
  INTEGER :: nb
END TYPE typ1

TYPE (typ1), PARAMETER :: values_ref = typ1(1)
TYPE (typ1) :: t1


CONTAINS
  SUBROUTINE sub1
    IMPLICIT NONE
    INTEGER :: array1(10)
    array1(:) = t1%nb
    array1(:) = values_ref%nb
  END SUBROUTINE sub1
END MODULE mod1
