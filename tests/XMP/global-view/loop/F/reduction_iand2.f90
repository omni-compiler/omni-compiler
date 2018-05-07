program main
  include 'xmp_lib.h'
  integer,parameter:: N=10
  integer random_array(N**2), ans_val
  integer a(N,N), sa, result
  real tmp(N,N)
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(cyclic,cyclic) onto p
!$xmp align a(i,j) with t(i,j)

  call random_number(tmp)
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        random_array(l) = int(tmp(i,j) * 10000)
     end do
  end do

!$xmp loop (i,j) on t(i,j)
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        a(i,j) = random_array(l)
     enddo
  enddo
         
  ans_val = -1
  do i=1, N**2
     ans_val = iand(ans_val, random_array(i))
  enddo

  sa = -1
!$xmp loop (i,j) on t(i,j) reduction(iand: sa)
  do j=1, N
     do i=1, N
        sa = iand(sa, a(i,j))
     enddo
  enddo

  result = 0
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
