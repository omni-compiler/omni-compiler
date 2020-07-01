program test

!$xmp nodes p(*)

!$xmp template t(64)
!$xmp distribute t(block) onto p

  integer :: a(64)
!$xmp align a(i) with t(i)
!$xmp shadow a(1)

  integer :: b(0:65)

  integer :: result = 0

!$xmp loop on t(i)
  do i = 1, 64
     a(i) = i
  end do

  b(0) = 64
  do i = 1, 64
     b(i) = i
  end do
  b(65) = 1

!$xmp reflect (a) width(/periodic/1:1)

!$xmp loop on t(i) reduction(+:result)
  do i = 1, 64

     if (a(i-1) /= b(i-1)) then
        result = 1
     end if

     if (a(i) /= b(i)) then
        result = 1
     end if

       if (a(i+1) /= b(i+1)) then
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
