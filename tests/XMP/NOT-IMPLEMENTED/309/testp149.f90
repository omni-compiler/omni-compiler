program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,*)
  integer aa(N), a
  real*4 bb(N), b
  real*8 cc(N), c
  integer procs, id, procs2, ans, result

  id = mod(xmp_node_num()-1, 4)+1
  procs = xmp_num_nodes()
  procs2 = procs/4
  result = 0

  a = xmp_node_num()
  b = real(a)
  c = dble(a)
  do i=1, N
     aa(i) = a+i-1
     bb(i) = real(a+i-1)
     cc(i) = dble(a+i-1)
  enddo

!$xmp bcast (a) on p(2:3,1:procs2)
!$xmp bcast (b) on p(2:3,1:procs2)
!$xmp bcast (c) on p(2:3,1:procs2)
!$xmp bcast (aa) on p(2:3,1:procs2)
!$xmp bcast (bb) on p(2:3,1:procs2)
!$xmp bcast (cc) on p(2:3,1:procs2)

  ans = 2
  if(id .ge. 2 .and. id .le. 3) then
     if(a .ne. ans) result = -1
     if(b .ne. real(ans)) result = -1
     if(c .ne. dble(ans)) result = -1
     do i=1, N
        if(aa(i) .ne. ans+i-1) result = -1
        if(bb(i) .ne. real(ans+i-1)) result = -1
        if(cc(i) .ne. dble(ans+i-1)) result = -1
     enddo
  else
     if(a .ne. xmp_node_num()) result = -1
     if(b .ne. real(a)) result = -1
     if(c .ne. dble(a)) result = -1
     do i=1, N
        if(aa(i) .ne. a+i-1) result = -1
        if(bb(i) .ne. real(a+i-1)) result = -1
        if(cc(i) .ne. dble(a+i-1)) result = -1
     enddo
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
