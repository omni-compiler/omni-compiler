!$xmp nodes p(4,4)

!$xmp template t(100,100)
!$xmp distribute t(block,cyclic(5)) onto p

  integer k

  k = 0

!$xmp task on p(2:3,1:4:2)
  k = 1
!$xmp end task

!$xmp task on p(2:3,1:4:2) nocomm
  k = 0
!$xmp end task

!$xmp reduction (+:k)

!$xmp task on p(1,1)
  if (k /= 0) then
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

  k = 0

!$xmp task on t(1,1)
  k = 1
!$xmp end task

!$xmp task on t(1,1) nocomm
  k = 0
!$xmp end task

!$xmp reduction (+:k)

!$xmp task on p(1,1)
  if (k /= 0) then
     write(*,*) "ERROR"
     call exit(1)
  else
     write(*,*) "PASS"
  end if
!$xmp end task

end program
