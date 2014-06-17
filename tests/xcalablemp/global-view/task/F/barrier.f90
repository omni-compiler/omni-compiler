program main
  include 'xmp_lib.h'
!$xmp nodes p(*)

  if(xmp_num_nodes().lt.4) then
     print *, 'You have to run this program by more than 3 nodes.'
     call exit(1)
  endif

!$xmp task on p(1:3)
!$xmp barrier
!$xmp end task

!$xmp barrier

!$xmp task on p(1)      
  write(*,*) "PASS"
!$xmp end task
end program main

