module mod1
implicit none

interface
  subroutine signal_handler()
  end subroutine
end interface

  procedure(signal_handler), public :: sigh1
  procedure(signal_handler), private :: sigh2

contains

  subroutine sub1(signalh)
    procedure(signal_handler), optional :: signalh
    procedure(signal_handler), pointer :: ptr
  end subroutine sub1

  subroutine sub2(signalh)
    procedure(signal_handler), pointer, intent(in) :: signalh
    procedure(signal_handler), pointer, save :: ptr2 => NULL()
  end subroutine sub2

end module mod1

