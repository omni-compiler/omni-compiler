program main
   integer, pointer, dimension(:) :: a
   integer :: x = 1
   if ( x .eq. 1 ) nullify (a)
end program main
