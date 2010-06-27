program main
  integer, pointer :: a
  integer, target  :: x = 1
  a=>x
  if ( x .eq. 1 ) nullify (a)
end program main
