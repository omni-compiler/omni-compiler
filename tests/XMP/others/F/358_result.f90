program test
  real(8) s, foo
  s = foo()
  if (s == 1.0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
end program test

function foo() result(rtime)
  real(8) rtime
  rtime = 1.0
  return
end function foo

