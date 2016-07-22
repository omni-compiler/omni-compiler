program xacc_reduction
  implicit none
  integer,parameter :: N = 128
  integer x, sum, sum2
  !$xmp nodes p(*)
  !$xmp template t(N)
  !$xmp distribute t(block) onto p

  sum = 0
  sum2 = 0

  !$acc data copy(sum) copyin(sum2)

  !$xmp loop (x) on t(x)
  !$acc parallel loop reduction(+:sum)
  do x = 1, N
     sum = sum + (x + 1)
     sum2 = sum2 + (x + 1)
  end do

  !$xmp reduction(+:sum) acc

  !$acc end data

  !$xmp task on p(1)
  if(sum == N*(N+1)/2+N .and. sum2 == 0)then
     print *, "OK"
  else
     print *, "invalid result", "sum=", sum, "sum2=", sum2
  end if
  !$xmp end task
end program xacc_reduction
