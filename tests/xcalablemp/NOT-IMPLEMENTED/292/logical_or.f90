program main
  include 'xmp_lib.h'
  integer,parameter:: N=4
  integer result
  logical sa
!$xmp nodes p(*)

  sa = .FALSE.
!$xmp reduction(.or.:sa)

  if( sa .neqv. .FALSE.) then
     result = -1
  endif

!$xmp task on p(1)
  sa = .TRUE.
!$xmp end task

!$xmp reduction(.or.:sa)

  if( sa .neqv. .TRUE.) then
     result = -1
  endif

  sa = .TRUE.
!$xmp reduction(.or.:sa)

  if( sa .neqv. .TRUE.) then
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
