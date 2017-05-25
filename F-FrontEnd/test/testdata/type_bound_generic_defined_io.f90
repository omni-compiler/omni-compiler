module tpb_defined_io_mod
  type node
     integer :: i
     real :: a
   contains
     procedure :: my_read_routine_formatted, my_read_routine_unformatted
     procedure :: my_write_routine_formatted, my_write_routine_unformatted
     generic :: READ(FORMATTED) => my_read_routine_formatted
     generic :: READ(UNFORMATTED) => my_read_routine_unformatted
     generic :: WRITE(FORMATTED) => my_write_routine_formatted
     generic :: WRITE(UNFORMATTED) => my_write_routine_unformatted
  end type node

contains
  SUBROUTINE my_read_routine_formatted (dtv, unit, iotype, v_list, iostat, iomsg)
    class(node) , INTENT(INOUT) :: dtv
    INTEGER, INTENT(IN) :: unit
    CHARACTER (LEN=*), INTENT(IN) :: iotype
    INTEGER, INTENT(IN) :: v_list(:)
    INTEGER, INTENT(OUT) :: iostat
    CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
  END SUBROUTINE my_read_routine_formatted
  SUBROUTINE my_read_routine_unformatted (dtv, unit, iostat, iomsg)
    class(node) , INTENT(INOUT) :: dtv
    INTEGER, INTENT(IN) :: unit
    INTEGER, INTENT(OUT) :: iostat
    CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
  END SUBROUTINE my_read_routine_unformatted
  SUBROUTINE my_write_routine_formatted(dtv, unit, iotype, v_list, iostat, iomsg)
    class(node), intent(in) :: dtv
    integer, intent(in) :: unit
    character (len=*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character (len=*), intent(inout) :: iomsg
    write (unit, '(dt)', iostat=iostat) dtv%i
  END SUBROUTINE my_write_routine_formatted
  SUBROUTINE my_write_routine_unformatted (dtv, unit, iostat, iomsg)
    class(node) , INTENT(IN) :: dtv
    INTEGER, INTENT(IN) :: unit
    INTEGER, INTENT(OUT) :: iostat
    CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
  END SUBROUTINE my_write_routine_unformatted
end module tpb_defined_io_mod

program test
  use tpb_defined_io_mod
  type(node) :: tt
  tt%i = 3
  tt%a = 2.0
  write(*, *) tt
end program test

