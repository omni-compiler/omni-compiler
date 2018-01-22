program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N,N)
!$xmp distribute t(*,block) onto p
  integer aa(N), a
  real*4 bb(N), b
  real*8 cc(N), c
  integer procs, ans, w, result

  procs = xmp_num_nodes()
  result = 0
  if(mod(N, procs) .eq. 0) then
     w = N/procs
  else
     w = N/procs+1
  endif

  do j=1, N
     a = xmp_node_num()
     b = real(a)
     c = dble(a)
     do i=1, N
        aa(i) = a+i-1
        bb(i) = real(a+i-1)
        cc(i) = dble(a+i-1)
     enddo

!$xmp bcast (a) from t(:,j) on p(:)
!$xmp bcast (b) from t(:,j) on p(:)
!$xmp bcast (c) from t(:,j) on p(:)
!$xmp bcast (aa) from t(:,j) on p(:)
!$xmp bcast (bb) from t(:,j) on p(:)
!$xmp bcast (cc) from t(:,j) on p(:)

     ans = (j-1)/w+1
     if(a .ne. ans) result = -1
     if(b .ne. real(ans)) result = -1
     if(c .ne. dble(ans)) result = -1
     do i=1, N
        if(aa(i) .ne. ans+i-1) result = -1
        if(bb(i) .ne. real(ans+i-1)) result = -1
        if(cc(i) .ne. dble(ans+i-1)) result = -1
     enddo
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
