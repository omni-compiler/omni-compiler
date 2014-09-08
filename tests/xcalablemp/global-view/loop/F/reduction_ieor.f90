program main
  include 'xmp_lib.h'
  integer, parameter:: N=1000
  integer random_array(N), ans_val, result
  integer a(N), sa
  real tmp(N)
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
!$xmp align a(i) with t(i)
  
  result = 0
  call random_number(tmp)
  random_array(:) = int(tmp(:) * 10000)

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = random_array(i)
  enddo
         
  ans_val = 0
  do i=1, N
     ans_val = ieor(ans_val, random_array(i))
  enddo

  sa = 0
!$xmp loop (i) on t(i) reduction(ieor: sa)
  do i=1, N
     sa = ieor(sa, a(i))
  enddo

  if( sa .ne. ans_val) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task


end program main
