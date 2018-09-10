program main

  type, bind(c) :: test
     real v
     real(8) w, x
  end type test

  !$xmp nodes p(2,2)

  !$xmp template t(10,10)

  !$xmp distribute t(block, block) onto p
  type(test) a(10,10)
  !$xmp align a(i,j) with t(i,j)
  !$xmp shadow a(1,1)

  !$xmp reflect (a) async(1)
  !$xmp wait_async(1)

!$xmp task on p(1,1)
  write(*,*) "PASS"
!$xmp end task

end program main
