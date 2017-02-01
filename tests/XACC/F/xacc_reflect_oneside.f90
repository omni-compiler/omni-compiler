program xacc_reflect_oneside
  integer, parameter :: ARRAY_SIZE = 16
  !$xmp nodes p(4)
  !$xmp template t(ARRAY_SIZE)
  !$xmp distribute t(block) onto p

  integer :: array_l(ARRAY_SIZE), array_u(ARRAY_SIZE)

  !$xmp align (i) with t(i) :: array_l, array_u
  !$xmp shadow array_l(1:0)
  !$xmp shadow array_u(0:1)


  !$xmp loop (i) on t(i)
  do i = 1, ARRAY_SIZE
     array_l(i) = i * 3 + 5
     array_u(i) = i * 3 + 5
  end do

  !$acc data copy(array_l, array_u)
  !$xmp reflect (array_l, array_u) acc
  !$acc end data

  err = 0

  !$xmp loop (i) on t(i) reduction(max:err)
  do i = 2, ARRAY_SIZE
     if(array_l(i-1) /= (i-1) * 3 + 5) then
        err = 1
     end if
  end do

  if (err > 0) then
     stop
  end if

  !$xmp loop (i) on t(i) reduction(max:err)
  do i = 1, ARRAY_SIZE - 1
     if(array_u(i+1) /= (i+1) * 3 + 5) then
        err = 1
     end if
  end do

  if (err > 0) then
     stop
  end if

  !$xmp task on p(1)
  write(*,*) "OK"
  !$xmp end task

end program xacc_reflect_oneside
