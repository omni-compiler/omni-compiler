MODULE mod1

CONTAINS
  SUBROUTINE exchg_boundaries (kdim, var01)
    IMPLICIT NONE

    INTEGER (KIND=4), INTENT(IN) :: kdim(24)
    REAL (KIND=8), INTENT (INOUT) :: var01(kdim(1))

    TYPE :: pointerto3d
      REAL(KIND=8) :: p
    END TYPE pointerto3d

    IF (kdim(1) <= 0) THEN
    END IF

  END SUBROUTINE exchg_boundaries
END MODULE mod1
