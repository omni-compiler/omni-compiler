module mod1
  use issue563_mod, only: t_message

  implicit none
  private 

  interface t_message
  
  end interface t_message

  abstract interface
    FUNCTION Handler_interface(msg_in) RESULT(msg_out)
      IMPORT :: t_Message
      CLASS(t_Message), INTENT(in)    :: msg_in
      CLASS(t_Message), POINTER       :: msg_out
    end function Handler_interface
  end interface

end module mod1
