      MODULE m
        IMPLICIT NONE

        TYPE :: t
          INTEGER :: i
        CONTAINS
          PROCEDURE :: assign_t_from_int
          PROCEDURE :: equals_t_int
          GENERIC :: ASSIGNMENT(=) => assign_t_from_int
          GENERIC :: OPERATOR(==) => equals_t_int
        END TYPE t

      CONTAINS

        SUBROUTINE assign_t_from_int (me, i)
          IMPLICIT NONE
          CLASS(t), INTENT(OUT) :: me
          INTEGER, INTENT(IN) :: i
          me%i = i
        END SUBROUTINE assign_t_from_int

        LOGICAL FUNCTION equals_t_int (me, i)
          IMPLICIT NONE
          CLASS(t), INTENT(IN) :: me
          INTEGER, INTENT(IN) :: i
          equals_t_int = (me%i == i)
        END FUNCTION equals_t_int

      END MODULE m

      PROGRAM main
        USE m
        TYPE(t) :: o1
        o1 = 10
        IF (o1==10) THEN
          PRINT *,'OK'
        ELSE
          PRINT *,'NG'
        END IF
      END PROGRAM main

