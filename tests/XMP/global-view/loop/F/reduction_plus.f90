program main
  include 'xmp_lib.h'
  integer,parameter:: N=100
!$xmp nodes p(*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(cyclic,*,*) onto p
!$xmp distribute t2(*,cyclic,*) onto p
!$xmp distribute t3(*,*,cyclic) onto p
  integer a(N), sa
  real*4 b(N), sb
  real*8 c(N), sc
  integer,allocatable:: w(:)
  integer ans, procs,result
!$xmp align a(i) with t1(i,*,*)
!$xmp align b(i) with t2(*,i,*)
!$xmp align c(i) with t3(*,*,i)

!$xmp loop (i) on t1(i,:,:)
  do i=1, N
     a(i) = xmp_node_num()
  enddo

!$xmp loop (i) on t2(:,i,:)
  do i=1, N
     b(i) = dble(xmp_node_num())
  enddo

!$xmp loop (i) on t3(:,:,i)
  do i=1, N
     c(i) = real(xmp_node_num())
  enddo

  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop (i) on t1(i,:,:) reduction(+: sa)
  do i=1, N
     sa = sa+a(i)
  enddo
      
!$xmp loop (i) on t2(:,i,:) reduction(+: sb)
  do i=1, N
     sb = sb+b(i)
  enddo

!$xmp loop (i) on t3(:,:,i) reduction(+: sc)
  do i=1, N
     sc = sc+c(i)
  enddo
      
  procs = xmp_num_nodes()
  allocate(w(1:procs))
  if(mod(N,procs) .eq. 0) then
     w = N/procs
  else
     do i=1, procs
        if(i .le. mod(N,procs)) then
           w(i) = N/procs+1
        else
           w(i) = N/procs
        endif
     enddo
  endif

  ans = 0
  do i=1, procs
     ans = ans + i*w(i)
  enddo
  
  result = 0
  if(  sa .ne. ans .or. abs(sb-dble(ans)) .gt. 0.0000001 .or. abs(sb-real(ans)) .gt. 0.0001 ) then
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
  
  deallocate(w)

end program main
