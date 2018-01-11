module test
  implicit none
  integer,parameter :: hoge = 20
end module test

program main
  character(len=10) string
  string = STRING
  if ( string == "P A S S" ) then
     print *, "PASS"
  else
     print *, "ERROR"
     call exit(1)
  end if
end program main
