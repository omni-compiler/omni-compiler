module mod1
implicit none

interface
  subroutine signal_handler( sig )
    integer, intent(in) :: sig
  end subroutine
end interface

contains

  subroutine set_handler( sig, signalh )
    integer :: sig
    procedure(signal_handler), optional :: signalh
  
    if(present(signalh)) then
      print*,'present'
    else
      print*,'not present'
    endif
  end subroutine
end module mod1

