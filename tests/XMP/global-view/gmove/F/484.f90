program test

!$xmp nodes p(2)

!$xmp template t(-7:4)
!$xmp distribute t(block) onto p

  integer a(4), b(4)
!$xmp align (i) with t(i) :: a, b

  integer :: result = 0

!$xmp loop on t(i)
  do i = 1, 4
     a(i) = 777
     b(i) = i
  end do

!$xmp gmove
  a(2:4) = b(2:4)

!$xmp loop (i) on t(i) reduction (+:result)
  do i = 2, 4
     if (a(i) /= i) then
        result = 1
     end if
  end do

!$xmp task on p(1)
  if (result == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test
