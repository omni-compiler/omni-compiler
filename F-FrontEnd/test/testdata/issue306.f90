program prog1
  character(len=:), allocatable :: string1
  character(kind=4,len=:),allocatable :: string2
end program prog1

subroutine sub1(string3)
  character(kind=4,len=*) :: string3
end subroutine sub1
