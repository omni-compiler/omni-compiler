program test

  integer a(10), b(10)

!$xmp nodes p(2)
!$xmp template t(10)
!$xmp distribute t(block) onto p

!$xmp align (i) with t(i) :: a, b
!$xmp shadow a(*)
!$xmp shadow b(*)

  integer me, xmp_node_num
  integer result

  me = xmp_node_num()

  do i = 1, 10
     a(i) = 0
     b(i) = 0
  end do

!$xmp task on p(1)
  a( 1) = 1
  a( 2) = 2
  a( 3) = 3
  a( 4) = 4
  a( 5) = 5
!$xmp end task
!$xmp task on p(2)
  a( 6) = 6
  a( 7) = 7
  a( 8) = 8
  a( 9) = 9
  a(10) = 10
!$xmp end task

!$xmp loop on t(i)
  do i = 1, 10
     b(i) = i
  end do

  result = 0

  do i = 1, 10
     if (a(i) /= b(i)) result = -1
  end do

!$xmp reduction(+:result)

!$xmp task on p(1)
  if ( result == 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test
