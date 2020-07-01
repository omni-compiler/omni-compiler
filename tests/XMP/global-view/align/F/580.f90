program main
  !$xmp template t(10)
  !$xmp nodes p(2)
  !$xmp distribute t(block) onto p
  integer a(10)
  !$xmp align a(i) with t(i)

  !$xmp task on p(1)
  write(*,*) "PASS"
  !$xmp end task
end program main
