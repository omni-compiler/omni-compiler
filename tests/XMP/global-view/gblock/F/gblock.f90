program test

!$xmp nodes p(4)

  integer m(4) = (/ 10, 20, 30, 40 /)

!$xmp template t1(100)
!$xmp distribute t1(block) onto p
!$xmp template t2(100)
!$xmp distribute t2(gblock(m)) onto p
!$xmp template t3(100)
!$xmp distribute t3(cyclic) onto p
!$xmp template t4(100)
!$xmp distribute t4(cyclic(3)) onto p

  integer a1(100), a2(100), a3(100), a4(100)
!$xmp align a1(i) with t1(i)
!$xmp align a2(i) with t2(i)
!$xmp align a3(i) with t3(i)
!$xmp align a4(i) with t4(i)
!$xmp shadow a1(1:2)
!$xmp shadow a2(1:2)

!$xmp loop on t1(i)
  do i = 1, 100
     a1(i) = i
  end do

!$xmp loop on t2(i)
  do i = 1, 100
     a2(i) = i
  end do

!$xmp loop on t3(i)
  do i = 1, 100
     a3(i) = i
  end do

!$xmp loop on t4(i)
  do i = 1, 100
     a4(i) = i
  end do

!$xmp task on t1(20)
  if (a1(20) /= 20) then
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

!$xmp task on t2(20)
  if (a2(20) /= 20) then
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

!$xmp task on t3(20)
  if (a3(20) /= 20) then
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

!$xmp task on t4(20)
  if (a4(20) /= 20) then
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

!$xmp barrier

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

end program test
