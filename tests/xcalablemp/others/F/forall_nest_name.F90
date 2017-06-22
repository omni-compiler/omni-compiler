PROGRAM main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
  IMPLICIT NONE
  INTEGER, DIMENSION(3,3) :: A
  INTEGER, DIMENSION(3,3) :: B = RESHAPE((/9,8,7,6,5,4,3,2,1/), (/3,3/))
  INTEGER, DIMENSION(3,3) :: C = RESHAPE((/1,2,3,4,5,6,7,8,9/), (/3,3/))
  INTEGER :: i, j
! FORALL (INTEGER(KIND=8) :: i = 1:3, j = 1:3, i.eq.j) A(i, j) = B(i, j)
  FORALL (i = 1:3:1, j = 1:3:2, i.eq.j) A(i, j) = B(i, j)
  this1 : FORALL (i = 1:3)
    this2 : FORALL (j = 1:3, i.ne.j)
      A(i, j) = C(i, j)
    END FORALL this2
  END FORALL this1
! PRINT *, A(:,1)
! PRINT *, A(:,2)
! PRINT *, A(:,3)
  IF (A(1,1).eq.9.and.A(2,1).eq.2) then
    PRINT *, 'PASS'
  ELSE
    PRINT *, 'NG'
    CALL EXIT(1)
  END IF
#else
  PRINT *, 'SKIPPED'
#endif
END PROGRAM main
