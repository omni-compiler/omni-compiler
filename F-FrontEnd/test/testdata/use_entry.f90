program use_entry
  use mod_entry
  integer i
  i = func(1, 2, 3)
  print *, i
  i = func_entry(1, 2, 3)
  print *, i
end program use_entry
