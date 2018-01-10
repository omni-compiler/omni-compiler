! testp154.f
! loop指示文とreduction節 (-) のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
      integer a(N), sa
      real*4 b(N), sb
      real*8 c(N), sc
      integer,allocatable:: w(:)
      integer ans, procs
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)
      character(len=2) result

!$xmp loop (i) onto t(i)
      do i=1, N
         a(i) = xmp_node_num()
         b(i) = dble(xmp_node_num())
         c(i) = real(xmp_node_num())
      enddo

      sa = 0
      sb = 0.0
      sc = 0.0

!$xmp loop (i) onto t(i) reduction(-: sa)
      do i=1, N
         sa = sa-a(i)
         sb = sb-b(i)
         sc = sc-c(i)
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
         ans = ans - i*w(i)
      enddo

      result = 'OK'
      if(  sa .ne. ans .or.
     $     abs(sb-dble(ans)) .gt. 0.0000001 .or.
     $     abs(sb-real(ans)) .gt. 0.0001 ) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp154.f ', result
      deallocate(w)

      end
