program test

  !$xmp nodes p(2,2)
  !$xmp template t(8,8)
  !$xmp distribute t(block,block) onto p

  integer a(8,8)
  !$xmp align a(i,j) with t(i,j)
  !$xmp shadow a(1,1)

  integer b(8,8)
  !$xmp align b(i,j) with t(i,j)
  !$xmp shadow b(1,1)

  integer result = 0
  
  !$xmp array on t
  a = 1

  !$xmp reflect (a) width(/periodic/1,/periodic/1)

  !$xmp array on t
  b = 1

  !$xmp loop (i,j) on t(i,j) margin(/unbound/1,/unbound/1)
  do j = 1, 8
     do i = 1, 8
        b(i,j) = 1
     end do
  end do

  !$xmp loop (i,j) on t(i,j) expand(/unbound/1,/unbound/1) reduction(+:result)
  do j = 1, 8
     do i = 1, 8
        if (a(i,j) /= b(i,j)) result = 1
     end do
  end do

!$xmp task on p(1,1)
  if ( result /= 0 ) then
     write(*,*) "ERROR"
     call exit(1)
  endif

  write(*,*) "PASS"
!$xmp end task
  
end program test
