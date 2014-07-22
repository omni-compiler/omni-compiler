program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp distribute t1(block,block) onto p
  integer a(N,N), sa, result
!$xmp align a(i,j) with t1(i,j)

  sa = 1
!$xmp loop (i,j) on t1(i,j)
  do j=1, N
     do i=1, N
        if(i.eq.j .and. mod(i,100).eq.0) then
           a(i,j) = 2
        else
           a(i,j) = 1
        endif
     enddo
  enddo

!$xmp loop (i,j) on t1(i,j)
  do j=1, N
     do i=1, N
        sa = sa*a(i,j)
     enddo
  enddo
!$xmp reduction (*:sa)

  result = 0
  if(  sa .ne. 1024 ) then
     result = -1 ! NG
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
