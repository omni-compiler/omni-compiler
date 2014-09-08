program main
  include 'xmp_lib.h'
  integer,parameter:: N=4
  integer result
  logical a(N), sa
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
!$xmp align a(i) with t(i)

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = .FALSE.
  enddo

!$xmp loop (i) on t(i) reduction(.and.:sa)
  do i=1, N
     sa = a(i)
  end do

  if( sa .neqv. .FALSE.) then
     result = -1
  endif

!$xmp task on p(1)
  a(1) = .TRUE.
!$xmp end task

!$xmp loop (i) on t(i) reduction(.and.:sa)
  do i=1, N
     sa = a(i)
  end do

  if( sa .neqv. .FALSE.) then
     result = -1
  endif

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = .TRUE.
  enddo
  
!$xmp loop (i) on t(i) reduction(.and.:sa)
  do i=1, N
     sa = a(i)
  end do

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
