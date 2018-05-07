MODULE issue563_mod

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_msg

  TYPE t_msg
    CHARACTER(len=50) :: name = ''
  END TYPE t_msg

  INTERFACE t_msg
    PROCEDURE construct_msg
  END INTERFACE t_msg

CONTAINS

  FUNCTION construct_msg(name) RESULT(return_ptr)
    CHARACTER(len=*), INTENT(in) :: name
    TYPE(t_msg), POINTER :: return_ptr
  END FUNCTION construct_msg

END MODULE issue563_mod
