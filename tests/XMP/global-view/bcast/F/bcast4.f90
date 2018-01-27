program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(*)
  integer aa(N), a
  real*4 bb(N), b
  real*8 cc(N), c
  integer procs, result

  procs = xmp_num_nodes()
  result = 0

  j = 1
  a = xmp_node_num()
  b = real(a)
  c = dble(a)
  do i=1, N
     aa(i) = a+i-1
     bb(i) = real(a+i-1)
     cc(i) = dble(a+i-1)
  enddo

!$xmp bcast (a)
!$xmp bcast (b)
!$xmp bcast (c)
!$xmp bcast (aa)
!$xmp bcast (bb)
!$xmp bcast (cc)
      
  if(a .ne. j) result = -1
  if(b .ne. real(j)) result = -1
  if(c .ne. dble(j)) result = -1
  do i=1, N
     if(aa(i) .ne. j+i-1) result = -1
     if(bb(i) .ne. real(j+i-1)) result = -1
     if(cc(i) .ne. dble(j+i-1)) result = -1
  enddo

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
