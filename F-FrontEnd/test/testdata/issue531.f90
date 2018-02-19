MODULE mod1
    IMPLICIT NONE

    TYPE :: type1
    CONTAINS
        PROCEDURE :: construct => s_construct
        PROCEDURE :: destruct => s_destruct
    END TYPE type1

CONTAINS

    SUBROUTINE s_construct(this)
        CLASS(type1), INTENT(INOUT) :: this
    END SUBROUTINE s_construct

    SUBROUTINE sub1(this)
        CLASS(type1), INTENT(INOUT) :: this
        CALL this%construct()
        CALL this%destruct()
    END SUBROUTINE sub1

    SUBROUTINE s_destruct(this)
        CLASS(type1), INTENT(INOUT) :: this
    END SUBROUTINE s_destruct

END MODULE mod1
