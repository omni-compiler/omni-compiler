module  m
  implicit none
  private
  type :: t
    private
    integer :: v
  end type
  type(t) :: v = t(k)
end module m
