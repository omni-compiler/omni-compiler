MODULE mod1

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_Message


  INTERFACE t_Message
    PROCEDURE Construct_message
  END INTERFACE t_Message

  TYPE t_Message
    CHARACTER(len=50)      :: name = ''
  END TYPE t_Message
  ABSTRACT INTERFACE
    FUNCTION Handler_interface(msg_in) RESULT(msg_out)
      IMPORT :: t_Message
      CLASS(t_Message), INTENT(in)    :: msg_in
      CLASS(t_Message), POINTER       :: msg_out
    END FUNCTION Handler_interface
  END INTERFACE

CONTAINS

  FUNCTION Construct_message(name) RESULT(return_ptr)

    CHARACTER(len=*), INTENT(in) :: name
    TYPE(t_Message),  POINTER    :: return_ptr

    ALLOCATE(return_ptr)
    return_ptr%name = TRIM(name)
  END FUNCTION Construct_message

END MODULE mod1
