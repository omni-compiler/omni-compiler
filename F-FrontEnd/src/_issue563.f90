module issue563_mod
  
  implicit none
  private

  public :: t_message

TYPE t_Message
    CHARACTER(len=50)      :: name = ''
  END TYPE t_Message 


end module issue563_mod
