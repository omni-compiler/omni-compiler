module mod
  type node
     integer :: i
     real :: a
  end type node
  INTERFACE READ(FORMATTED)
    SUBROUTINE my_read_routine_formatted (dtv, unit, iotype, v_list, iostat, iomsg)
      class(node) , INTENT(INOUT) :: dtv
      INTEGER, INTENT(IN) :: unit
      CHARACTER (LEN=*), INTENT(IN) :: iotype
      INTEGER, INTENT(IN) :: v_list(:)
      INTEGER, INTENT(OUT) :: iostat
      CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
    END SUBROUTINE my_read_routine_formatted
  END INTERFACE READ(FORMATTED)
  INTERFACE READ(UNFORMATTED)
     SUBROUTINE my_read_routine_unformatted (dtv, unit, iostat, iomsg)
       class(node) , INTENT(INOUT) :: dtv
       INTEGER, INTENT(IN) :: unit
       INTEGER, INTENT(OUT) :: iostat
       CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     END SUBROUTINE my_read_routine_unformatted
  END INTERFACE READ(UNFORMATTED)
  INTERFACE WRITE(FORMATTED)
     SUBROUTINE my_write_routine_formatted(dtv, unit, iotype, v_list, iostat, iomsg)
       class(node), intent(in) :: dtv
       integer, intent(in) :: unit
       character (len=*), intent(in) :: iotype
       integer, intent(in) :: v_list(:)
       integer, intent(out) :: iostat
       character (len=*), intent(inout) :: iomsg
       write (unit, '(dt)', iostat=iostat) dtv%i
     END SUBROUTINE my_write_routine_formatted
  END INTERFACE WRITE(FORMATTED)
  INTERFACE WRITE(UNFORMATTED)
     SUBROUTINE my_write_routine_unformatted (dtv, unit, iostat, iomsg)
       class(node) , INTENT(IN) :: dtv
       INTEGER, INTENT(IN) :: unit
       INTEGER, INTENT(OUT) :: iostat
       CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
     END SUBROUTINE my_write_routine_unformatted
  END INTERFACE WRITE(UNFORMATTED)
end module mod

program test
  use mod
  type(node) :: tt
  tt%i = 3
  tt%a = 2.0
  write(*, *) tt
end program test

