MODULE mod1

  USE issue563_mod
!  USE issue563_mod, ONLY: t_msg

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
END MODULE mod1
