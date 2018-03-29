MODULE mod1

  USE issue563_mod, ONLY: t_msg

  IMPLICIT NONE
  PRIVATE

  ABSTRACT INTERFACE
    FUNCTION handler(this, msg_in) RESULT(return_ptr)
      IMPORT :: t_msg
      CLASS(t_msg), INTENT(inout) :: this
      CLASS(t_msg), INTENT(in)    :: msg_in
      CLASS(t_msg), POINTER       :: return_ptr
    END FUNCTION handler
  END INTERFACE

  TYPE t_state
  END TYPE t_state

  TYPE, ABSTRACT, EXTENDS(t_state) :: t_upstate
    TYPE(t_msg), ALLOCATABLE :: ts(:)
  END TYPE t_upstate

END MODULE mod1
