program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N,N)
  integer procs, result
  integer a(N,N), sa, ans
  real*8  b(N,N), sb
  real*4  c(N,N), sc
  integer :: g(4) = (/50,50,50,850/)
!$xmp distribute t(gblock(g),block) onto p
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)

  sa = 0
  sb = 0.0
  sc = 0.0
      
!$xmp loop (i,j) on t(i,j)
  do j=1, N
     do i=1, N
        a(i,j) = 1
        b(i,j) = 2.0
        c(i,j) = 3.0
     enddo
  enddo
  
!$xmp loop (i,j) on t(i,j)
  do j=1, N
     do i=1, N
        sa = sa + a(i,j)
        sb = sb + b(i,j)
        sc = sc + c(i,j)
     enddo
  enddo

  result = 0
  procs = xmp_node_num()
  ans = g( (mod((procs)-1,4)+1)) * (1000/4)

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
!$xmp task on p(1,1)
  if (result .eq. 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  end if
!$xmp end task

end program main

         
