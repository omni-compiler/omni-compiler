program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,*)
  integer aa(N), a
  real*4 bb(N), b
  real*8 cc(N), c
  integer procs, id, procs1, procs2, ans, result

  id = xmp_node_num()
  procs = xmp_num_nodes()
  procs1 = mod(id,4)
  procs2 = procs/4
  result = 0

  do k=1,procs2
     do j=2, 3
        a = xmp_node_num()
        b = real(a)
        c = dble(a)
        do i=1, N
           aa(i) = a+i-1
           bb(i) = real(a+i-1)
           cc(i) = dble(a+i-1)
        enddo
            
!$xmp bcast (a) from p(j,k) on p(2:3,1:procs2) async(100)
!$xmp bcast (b) from p(j,k) on p(2:3,1:procs2) async(100)
!$xmp bcast (c) from p(j,k) on p(2:3,1:procs2) async(200)
!$xmp bcast (aa) from p(j,k) on p(2:3,1:procs2) async(200)
!$xmp bcast (bb) from p(j,k) on p(2:3,1:procs2) async(100)
!$xmp bcast (cc) from p(j,k) on p(2:3,1:procs2) async(200)

!$xmp wait_async(100,200) on p(2:3,1:procs2)

        ans = (k-1)*4+j
        if(procs1 .ge. 2 .and. procs1 .le. 3) then
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
     enddo
  enddo
  
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
