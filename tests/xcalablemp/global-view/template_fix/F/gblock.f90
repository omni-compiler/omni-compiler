program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(gblock(*)) onto p
  integer s, procs, remain, result
  integer,allocatable:: a(:)
  integer,allocatable:: m(:)
!$xmp align a(i) with t(i)
  
  procs = xmp_num_nodes()
  allocate(m(procs))
      
  remain = N
  do i=1, procs-1
     m(i) = remain/2
     remain = remain-m(i)
  enddo
  m(procs) = remain

!$xmp template_fix (gblock(m)) t(N)
  allocate(a(N))

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = i
  enddo
  
  s = 0
!$xmp loop (i) on t(i) reduction(+: s)
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

  deallocate(a, m)

end program main
