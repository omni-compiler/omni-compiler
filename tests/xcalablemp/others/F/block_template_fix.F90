program main
  include 'xmp_lib.h'
!$xmp nodes p(*)

#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

!$xmp template t(:)
!$xmp distribute t(block) onto p
  integer N, s, result
  integer,allocatable:: a(:)
!$xmp align a(i) with t(i)
  
block
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
end block

#else
!$xmp task on p(1)
  print *, 'SKIPPED'
!$xmp end task
#endif

end program main
