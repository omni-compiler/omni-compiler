MODULE mod1
  IMPLICIT NONE
CONTAINS
  SUBROUTINE sub1()
    IMPLICIT NONE

    REAL :: varLevel1

  CONTAINS
    SUBROUTINE subsub1()
      REAL :: varLevel2

      varLevel1 = 1.5
      varLevel2 = varLevel1 * fct1(1.2)
    END SUBROUTINE subsub1
  END SUBROUTINE sub1
END MODULE mod1
