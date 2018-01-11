program test

!$xmp nodes p(4)

!$xmp template t(100)
!$xmp distribute t(block) onto p

  integer x(100,100), y(100,100)
!$xmp align x(*,j) with t(j)

  integer :: result = 0

!$xmp array on t
  x = 0

!$xmp array on t(2:99)
  x(2:99, 2:99) = 1

  do j =1, 100
     y(:,j) = 0
  end do

  y(2:99, 2:99) = 1

!$xmp loop on t(j)
  do j = 1, 100
     do i = 1, 100
        if (x(i,j) /= y(i,j)) result = -1
     end do
  end do

!$xmp reduction(+:result)

!$xmp task on p(1)
  if ( result /= 0 ) then
     write(*,*) "ERROR"
     call exit(1)
  else
     write(*,*) "PASS"
  endif
!$xmp end task

end program test
