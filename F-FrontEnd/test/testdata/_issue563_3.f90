MODULE issue563_3_mod

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: t_action, t_hsm

  TYPE t_action
    CHARACTER(len=10) :: name = ""
  END TYPE t_action

  INTERFACE t_action
    PROCEDURE construct_action
  END INTERFACE t_action

  TYPE, ABSTRACT :: t_hsm
    CHARACTER(len=30)           :: name
    TYPE(t_Action),   ALLOCATABLE :: actions(:)         !< List of possible actions (events)
  CONTAINS
    PROCEDURE :: Get_action        => Get_action_hsm
  END TYPE t_hsm

CONTAINS

FUNCTION Construct_action(name) RESULT(return_ptr)

    CHARACTER(len=*), INTENT(in) :: name
    TYPE(t_Action), POINTER :: return_ptr

    ALLOCATE(return_ptr)
    return_ptr%name = TRIM(name)

  END FUNCTION Construct_action

  FUNCTION Get_action_hsm(this, name) RESULT(return_value)

    CLASS(t_Hsm),     TARGET, INTENT(in) :: this
    CHARACTER(len=*),         INTENT(in) :: name
    TYPE(t_Action)                       :: return_value

  END FUNCTION Get_action_hsm

END MODULE issue563_3_mod
