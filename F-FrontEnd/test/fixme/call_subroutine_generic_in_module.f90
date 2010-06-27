module m
  implicit none
  private
  public :: s
  interface g
    module procedure s
  end interface
contains
  subroutine s()
  end subroutine s
end module m

module m2
implicit none
private
public :: g0
interface g0
  module procedure g1
end interface
contains
subroutine g1
  use m
  call g()
end subroutine
end module m2
