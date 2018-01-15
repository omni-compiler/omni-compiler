program main
  include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(block) onto p
  integer N, s, result
  integer,allocatable:: a(:)
!$xmp align a(i) with t(i)
  
  N = 1000
!$xmp template_fix (block) t(N)
  allocate(a(N))

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = i
  enddo
         
  s = 0
!$xmp loop (i) on t(i) reduction(+:s)
  do i=1, N
     s = s+a(i)
  enddo
  
  result = 0
  if(s .ne. 500500) then
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

  deallocate(a)

end program main
