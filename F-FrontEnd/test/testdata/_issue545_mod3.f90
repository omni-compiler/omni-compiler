module issue545_mod3
  use issue545_mod2, only: Type1

  implicit none
  private

  public :: type2

  type :: type2
    class(Type1), pointer :: p
  end type

end module issue545_mod3
