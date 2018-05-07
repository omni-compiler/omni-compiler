#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM PROC_PTR_EXAMPLE
        REAL :: R1
        INTEGER :: I1
        INTERFACE
           SUBROUTINE SUB(X)
             INTEGER, INTENT(IN) :: X
           END SUBROUTINE SUB
           FUNCTION REAL_FUNC(Y)
             REAL, INTENT(IN) :: Y
             REAL :: REAL_FUNC
           END FUNCTION REAL_FUNC
           FUNCTION INT_FUNC(X, Y)
             INTEGER, INTENT(IN) :: X, Y
             INTEGER :: INT_FUNC
           END FUNCTION INT_FUNC
        END INTERFACE
        ! with explicit interface
        PROCEDURE(SUB), POINTER :: PTR_TO_SUB
        ! with explicit interface
        PROCEDURE(REAL_FUNC), POINTER :: PTR_TO_REAL_FUNC => NULL()
        ! with implicit interface
        PROCEDURE(INTEGER), POINTER :: PTR_TO_INT
        PTR_TO_SUB => SUB
        PTR_TO_REAL_FUNC => REAL_FUNC
        PTR_TO_INT => INT_FUNC
        I1 = 1
        CALL PTR_TO_SUB(I1)
        R1 = PTR_TO_REAL_FUNC(3.0)
        I2 = PTR_TO_INT(4, 5) + I1 + R1
        if(I2.eq.15) then
          PRINT *, 'PASS ALL'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      END PROGRAM PROC_PTR_EXAMPLE

      SUBROUTINE SUB(X)
        INTEGER, INTENT(INOUT) :: X
        PRINT *, 'PASS 1'
        X = X + 2
      END SUBROUTINE SUB

      FUNCTION REAL_FUNC(Y)
        REAL, INTENT(IN) :: Y
        REAL :: REAL_FUNC
        PRINT *, 'PASS 2'
        REAL_FUNC = Y
      END FUNCTION REAL_FUNC

      FUNCTION INT_FUNC(X, Y)
        INTEGER, INTENT(IN) :: X, Y
        INTEGER :: INT_FUNC
        PRINT *, 'PASS 3'
        INT_FUNC = X + Y
      END FUNCTION INT_FUNC
#else
PRINT *, 'SKIPPED'
END
#endif
