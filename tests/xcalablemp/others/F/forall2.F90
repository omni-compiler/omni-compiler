PROGRAM main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
  IMPLICIT NONE
  INTEGER, DIMENSION(3,3) :: A
  INTEGER, DIMENSION(3,3) :: B = RESHAPE((/1,2,3,4,5,6,7,8,9/), (/3,3/))
  FORALL (INTEGER(KIND=8) :: i = 1:3, j = 1:3, .TRUE.)
    A(i, j) = B(i, j)
  END FORALL
! PRINT *, A(:,1)
! PRINT *, A(:,2)
! PRINT *, A(:,3)
  IF (A(2,1).eq.2) then
    PRINT *, 'PASS'
  ELSE
    PRINT *, 'NG'
    CALL EXIT(1)
  END IF
#else
PRINT *, 'SKIPPED'
#endif
      END PROGRAM main
      
