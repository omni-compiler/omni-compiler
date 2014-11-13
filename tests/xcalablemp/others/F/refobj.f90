program refobj
  real*8 t0
!$xmp nodes p(*)
!$xmp template t(100)
!$xmp distribute t(block) onto p

  t0 = 0.0d0

  !$xmp loop on t(i)
  do i = 1, 100
     continue
  end do

  write(*,931) t0
931 format(1X,F16.3)

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

end program refobj
