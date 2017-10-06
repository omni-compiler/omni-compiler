module abstract_m
  type typ1
  contains
    procedure, private :: assign0
    generic :: dummy => assign0
    generic, public :: assign => assign0
    generic, private :: assignp => assign0
  end type
  interface
    subroutine assign0(this)
      import typ1
      class(typ1) ,intent(in) :: this
    end subroutine
  end interface
end module

