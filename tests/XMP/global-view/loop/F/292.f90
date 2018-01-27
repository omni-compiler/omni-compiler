program main
  !$xmp nodes p(2)
  !$xmp template t(10)
  !$xmp distribute t(block) onto p
  integer a(10), sa, ia
  !$xmp align a(i) with t(i)

  !$xmp task on p(1)
  a(1) = 2
  a(2) = 1
  a(3) = 4
  a(4) = 6
  a(5) = 3
  !$xmp end task

  !$xmp task on p(2)
  a(6)  = 2
  a(7)  = 6
  a(8)  = 3
  a(9)  = 3
  a(10) = 1
  !$xmp end task

  ! FIRST MAX
  sa = 0
  !$xmp loop (i) on t(i) reduction(firstmax:sa/ia/)
  do i=1, 10
     if(sa .lt. a(i)) then
        ia = i
        sa = a(i)
     endif
  enddo

  !$xmp task on p(1)
  if(ia == 4 .and. sa == 6) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
  !$xmp end task
  
  ! LAST MAX
   sa = 0
  !$xmp loop (i) on t(i) reduction(lastmax:sa/ia/)
  do i=1, 10
     if(sa .lt. a(i)) then
        ia = i
        sa = a(i)
     endif
  enddo
  
  !$xmp task on p(1)  
  if(ia == 7 .and. sa == 6) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
  !$xmp end task


  ! FIRST MIN
  sa = 10
  !$xmp loop (i) on t(i) reduction(firstmin:sa/ia/)
  do i=1, 10
     if(sa .gt. a(i)) then
        ia = i
        sa = a(i)
     endif
  enddo

  !$xmp task on p(1)
  if(ia == 2 .and. sa == 1) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
  !$xmp end task

  
  ! LAST MIN
  sa = 10
  !$xmp loop (i) on t(i) reduction(lastmin:sa/ia/)
  do i=1, 10
     if(sa .gt. a(i)) then
        ia = i
        sa = a(i)
     endif
  enddo

  !$xmp task on p(1)
  if(ia == 10 .and. sa == 1) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
    !$xmp end task
end program main
