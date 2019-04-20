program test_loc
  logical pp
  real a
  if (loc(a).ne.loc(pp)) then
    ierr=0
  else
    ierr=1
  endif
  call chk_int(ierr)
end program

