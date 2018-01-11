program main
  use init
  !$xmp nodes p(2)
  !$xmp template t(32)
  !$xmp distribute t(block) onto p
  implicit none
  complex(8) ::  ue_t(16,32)
  !$xmp align  ue_t(*,k) with t(k)
  !$xmp shadow ue_t(0,1)

  call init_u_and_y(ue_t)

  !$xmp task on p(1)
  write(*,*) 'PASS'
  !$xmp end task
end program main
