program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
  integer random_array(N), ans_val, val, result
  real tmp(N)
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p

  result = 0
  call random_number(tmp)
  random_array(:) = int(tmp(:) * 10000)

  ans_val = 2147483647
  do i=1, N
     ans_val = min(ans_val, random_array(i))
  enddo

  val = 2147483647
!$xmp loop on t(i)
  do i=1, N
     val = min(val, random_array(i))
  enddo
!$xmp reduction(min: val)
  if(val .ne. ans_val) then
     result = -1  ! NG
  endif

!$xmp reduction(+: result)
!$xmp task on p(1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program main      
