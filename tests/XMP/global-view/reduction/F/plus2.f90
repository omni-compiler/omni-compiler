program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(block,cyclic) onto p
!$xmp distribute t3(cyclic,cyclic) onto p
  integer a(N,N), sa
  real*8  b(N,N), sb
  real*4  c(N,N), sc
  integer result
!$xmp align a(i,j) with t1(i,j)
!$xmp align b(i,j) with t2(i,j)
!$xmp align c(i,j) with t3(i,j)

  if(xmp_num_nodes().lt.4) then
     print *, 'You have to run this program by more than 4 nodes.'
     call exit(1)
  endif

  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop (i,j) on t1(i,j)
  do j=1, N
     do i=1, N
        a(i,j) = 1
     enddo
  enddo

!$xmp loop (i,j) on t2(i,j)
  do j=1, N
     do i=1, N
        b(i,j) = 0.5
     enddo
  enddo

!$xmp loop (i,j) on t3(i,j)
  do j=1, N
     do i=1, N
        c(i,j) = 0.25
     enddo
  enddo

!$xmp loop (i,j) on t1(i,j)
  do j=1, N
     do i=1, N
        sa = sa+a(i,j)
     enddo
  enddo

!$xmp loop (i,j) on t2(i,j)
  do j=1, N
     do i=1, N
        sb = sb+b(i,j)
     enddo
  enddo

!$xmp loop (i,j) on t3(i,j)
  do j=1, N
     do i=1, N
        sc = sc+c(i,j)
     enddo
  enddo

!$xmp reduction (+:sa)
!$xmp reduction (+:sb)
!$xmp reduction (+:sc)
      
  result = 0
  if( sa .ne. N**2 .or. abs(sb-(dble(N**2)*0.5)) .gt. 0.000001 .or. abs(sc-(real(N**2)*0.25)) .gt. 0.0001) then
     result = -1 ! ERROR
     print *, sa, sb, sc
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
