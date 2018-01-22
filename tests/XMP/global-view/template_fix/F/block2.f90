program main
  integer i
  integer, parameter :: x_min = 1, x_max = 16
  integer, allocatable :: a(:)
  !$xmp nodes p(*)
  !$xmp template t(:)
  !$xmp distribute t(block) onto p
  !$xmp align a(i) with t(i)

  !$xmp template_fix(block) t(x_min-1: x_max+1)

  allocate(a(x_min:x_max))

  !$xmp loop (i) on t(i)
  do i = x_min, x_max
    a(i) = i
  end do

  !$xmp task on t(1)
  if( a(1) .eq. 1 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
  !$xmp end task

  deallocate(a)

end program
