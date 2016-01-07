program test

!$xmp nodes p(2,2)

!$xmp template t(4,4)
!$xmp distribute t(block,block) onto p

  integer a(4,4)
!$xmp align a(i,j) with t(i,j)
!$xmp shadow a(*,*)

  integer :: result = 0

!$xmp loop (i,j) on t(i,j)
  do j = 1, 4
     do i = 1, 4
        a(i,j) = i * 10 + j
     end do
  end do

!$xmp reflect (a)

  do j = 1, 4
     do i = 1, 4
        if (a(i,j) /= i * 10 + j) then
           result = 1
        end if
     end do
  end do

!$xmp reduction (+:result)

!$xmp task on p(1,1)
  if (result == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test
