program xacc_reflect
  implicit none
  integer,parameter :: XSIZE = 16
  integer a(XSIZE), b(XSIZE), x
  logical :: flag = .true.
  !$xmp nodes p(*)
  !$xmp template t(XSIZE)
  !$xmp distribute t(block) onto p
  !$xmp align (x) with t(x) :: a, b
  !$xmp shadow a(1)

  !$acc data copy(a)

  !$xmp loop (x) on t(x)
  !$acc parallel loop
  do x = 1, XSIZE
     a(x) = x
  end do

  !$xmp reflect(a) acc

  !$xmp loop (x) on t(x)
  !$acc parallel loop
  do x = 2, XSIZE-1
     b(x) = a(x-1) + a(x+1)
  end do

  !$acc end data

  !check
  !$xmp loop (x) on t(x)
  do x = 2, XSIZE-1
     if(b(x) .ne. x * 2) then
        flag = .false.
     end if
  end do
  !$xmp reduction(.and.:flag)

  !$xmp task on p(1)
  if(flag) then
     print *, "OK"
  else
     print *, "invalid result"
  end if
  !$xmp end task
end program xacc_reflect
