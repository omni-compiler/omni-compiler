module m
  character(len=6), parameter :: s = "aaaaaa"
end module m

module n
  integer :: s
end module n

program main
  use m
  use n 
end program main
