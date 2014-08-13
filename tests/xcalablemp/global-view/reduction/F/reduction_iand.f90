program main
  include 'xmp_lib.h'
  integer,parameter:: N=10
  integer random_array(N), ans_val
  integer a(N), sa, result
  real tmp(N)
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
!$xmp align a(i) with t(i)

  call random_number( tmp )
  random_array(:) = int(tmp(:) * 10000)

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = random_array(i)
  enddo
         
  ans_val = -1
  do i=1, N
     ans_val = iand(ans_val, random_array(i))
  enddo

  sa = -1
!$xmp loop (i) on t(i) 
  do i=1, N
     sa = iand(sa, a(i))
  enddo
!$xmp reduction(iand: sa)

  result = 0
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
