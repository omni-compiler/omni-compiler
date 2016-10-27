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
          TYPE(t), INTENT(OUT) :: me
          INTEGER, INTENT(IN) :: i
          me = t (i)
        END SUBROUTINE assign_t_from_int

        LOGICAL FUNCTION equals_t_int (me, i)
          IMPLICIT NONE
          TYPE(t) :: me
          INTEGER :: i
          equals_t_int = (me%i == i)
        END FUNCTION equals_t_int

      END MODULE m
