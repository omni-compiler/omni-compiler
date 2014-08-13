program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t(N)
  integer a(N), sa, ans, result, proc
  real*8  b(N), sb
  real*4  c(N), sc
  integer :: g(4) = (/ 100, 200, 600, 100 /)
!$xmp distribute t(gblock(g)) onto p
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)

  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop on t(j)
  do j=1, N
     a(j) = 1
     b(j) = 2.0
     c(j) = 3.0
  enddo
      
!$xmp loop on t(j)
  do j=1, N
     sa = sa + a(j)
     sb = sb + b(j)
     sc = sc + c(j)
  enddo
  
  result = 0
  proc = xmp_node_num()
  ans = g(proc)

  if(sa.ne.ans) then
     result = -1
  endif
  if(abs(sb-2.0*dble(ans)).gt.0.000000001) then
     result = -1
  endif
  if(abs(sc-3*real(ans)).gt.0.000001) then
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
