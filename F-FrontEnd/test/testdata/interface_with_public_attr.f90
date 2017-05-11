module test
  implicit none
  public :: hoge

  interface hoge
    module procedure foo
  end interface

contains
  subroutine foo()
  end subroutine

end module
