module issue319

  public :: main

  type typ1
    contains
      procedure, nopass, public :: id    
  end type typ1

  type(typ1), save :: main

contains
  function id()
    integer :: id
  end function id

end module issue319
