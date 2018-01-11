program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
  integer a(N), sa, result
  real*4 b(N), sb
  real*8 c(N), sc
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)

!$xmp loop (i) on t(i)
  do i=1, N
     if(mod(i,100).eq.0) then
        a(i) = 2
        b(i) = 2.0
        c(i) = 4.0
     else if(mod(i,100).eq.50) then
        a(i) = 1.0
        b(i) = 0.5
        c(i) = 0.25
     else
        a(i) = 1
        b(i) = 1.0
        c(i) = 1.0
     endif
  enddo

  sa = 1
  sb = 1.0
  sc = 1.0

!$xmp loop (i) on t(i) reduction(*: sa, sb, sc)
  do i=1, N
     sa = sa*a(i)
     sb = sb*b(i)
     sc = sc*c(i)
  enddo

  result = 0
  if( sa .ne. 1024 .or. abs(sb-dble(1.0)) .gt. 0.0000001 .or. abs(sb-real(1.0)) .gt. 0.0001 ) then
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
