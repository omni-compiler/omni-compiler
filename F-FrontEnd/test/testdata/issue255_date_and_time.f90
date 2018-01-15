module mod1
 
  integer, parameter :: iintegers = 4

  contains

  subroutine sub1()
    integer(kind=iintegers) :: dt(8)

    call date_and_time(values=dt)

  end subroutine sub1

end module mod1
