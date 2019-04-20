program main
  real,allocatable intreal(:)[:]
contains
  subroutine sub
    integer,allocatable intreal(:)[:]
    allocate(intreal(3)[*])
  end subroutine sub
end program main
