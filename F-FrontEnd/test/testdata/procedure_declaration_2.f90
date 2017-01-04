      PROGRAM PROC_PTR_EXAMPLE
        REAL :: R1
        INTEGER :: I1
        INTERFACE
           SUBROUTINE SUB(X)
             REAL, INTENT(IN) :: X
           END SUBROUTINE SUB
           FUNCTION REAL_FUNC(Y)
             REAL, INTENT(IN) :: Y
             REAL :: REAL_FUNC
           END FUNCTION REAL_FUNC
        END INTERFACE
        ! with explicit interface
        PROCEDURE(SUB), POINTER :: PTR_TO_SUB 
        ! with explicit interface
        PROCEDURE(REAL_FUNC), POINTER :: PTR_TO_REAL_FUNC => NULL()  
        ! with implicit interface
        PROCEDURE(INTEGER), POINTER :: PTR_TO_INT                    
        PTR_TO_SUB => SUB
        PTR_TO_REAL_FUNC => REAL_FUNC
        CALL PTR_TO_SUB(1.0)
        R1 = PTR_TO_REAL_FUNC(2.0)
        I1 = PTR_TO_INT(M, N)
      END PROGRAM PROC_PTR_EXAMPLE

      SUBROUTINE SUB(X)
        REAL, INTENT(IN) :: X
      END SUBROUTINE SUB

      FUNCTION REAL_FUNC(Y)
        REAL, INTENT(IN) :: Y
        REAL :: REAL_FUNC
      END FUNCTION REAL_FUNC
