module mod
  type node
     integer :: i
     real :: a
   contains
     procedure :: mywrite
     generic :: WRITE(FORMATTED) => mywrite
  end type node

contains
  subroutine mywrite(dtv, unit, iotype, vlist, iostat, iomsg)
    class(node), intent(in) :: dtv
    integer, intent(in) :: unit
    character (len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character (len=*), intent(inout) :: iomsg
    write (unit, '(dt)', iostat=iostat) dtv%i
  end subroutine mywrite
end module mod

program test
  use mode
  type(node) :: tt
  tt%i = 3
  tt%a = 2.0
  write(*, *) tt
end program test

