program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
  integer random_array(N*N), ans_val
  integer a(N,N), sa, result
  real tmp(N*N)
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(cyclic,cyclic) onto p
!$xmp align a(i,j) with t(i,j)

  result = 0
  call random_number(tmp)
  random_array(:) = int(tmp(:) * 10000)

!$xmp loop (i,j) on t(i,j)
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        a(i,j) = random_array(l)
     enddo
  enddo
         
  ans_val = 0
  do i=1, N**2
     ans_val = ieor(ans_val, random_array(i))
  enddo

  sa = 0
!$xmp loop (i,j) on t(i,j) reduction(ieor: sa)
  do j=1, N
     do i=1, N
        sa = ieor(sa, a(i,j))
     enddo
  enddo
         
  if( sa .ne. ans_val) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1,1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif

!$xmp end task

end program main
