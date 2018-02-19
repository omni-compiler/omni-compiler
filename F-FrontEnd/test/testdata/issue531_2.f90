MODULE mod1

    IMPLICIT NONE

PUBLIC :: type1, deleteType, destructType

    TYPE, ABSTRACT :: type1
    CONTAINS
        PROCEDURE :: delete => deleteType
        PROCEDURE :: destruct => destructType !< destructor
    END TYPE type1

PRIVATE

CONTAINS
    SUBROUTINE deleteType(this)
        CLASS(type1), POINTER, INTENT(INOUT) :: this
        CALL this%destruct()
        DEALLOCATE(this)
    END SUBROUTINE deleteType

    SUBROUTINE destructType(this)
        CLASS(type1), TARGET, INTENT(INOUT) :: this
    END SUBROUTINE destructType

END MODULE mod1
