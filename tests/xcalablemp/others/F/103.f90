program test103

  double precision :: s

!$xmp nodes p(2)

!$xmp template t(10)
!$xmp distribute t(block) onto p

  s = 0

!$xmp loop on t(i) reduction (+:s)
  do i = 1, 10
    s = s + i
 end do

!$xmp task on p(1) nocomm
 if (s == 55) then
    write(*,*) "PASS"
 else
    write(*,*) "ERROR"
    call exit(1)
 endif
!$xmp end task

end program test103
