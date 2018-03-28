MODULE mod1
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: t_state

  TYPE, ABSTRACT :: t_state
    CLASS(t_State), POINTER :: parent => NULL()
  CONTAINS
    PROCEDURE :: init_state
  END TYPE t_State

CONTAINS
  SUBROUTINE init_state(this, that)
    CLASS(t_state), INTENT(inout), TARGET :: this
    CLASS(t_state), POINTER, OPTIONAL :: that 

    this%parent => that

    SELECT TYPE(that)
      CLASS IS (t_state)
        this%parent => that
    END SELECT

  END SUBROUTINE init_state
END MODULE mod1
