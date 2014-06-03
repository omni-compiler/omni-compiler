! testp073.f
! task指示文とreduction指示文の組合せ
      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      integer procs, w
      integer a(N,N), sa, ans
      real*8  b(N,N), sb
      real*4  c(N,N), sc
!$xmp align a(*,i) with t(i)
!$xmp align b(*,i) with t(i)
!$xmp align c(*,i) with t(i)
!$xmp shadow a(0,1)
!$xmp shadow a(0,2)
!$xmp shadow a(0,3)
      character(len=2) result

      if(xmp_num_nodes().lt.2) then
         print *, 'You have to run this program by more than 2 nodes.'
         call exit(1)
      endif

      sa = 0
      sb = 0.0
      sc = 0.0

      procs = xmp_num_nodes()
      if(mod(N,procs).eq.0) then
         w = N/procs
      else
         w = N/procs+1
      endif
      
!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            a(i,j) = 1
            b(i,j) = 2.0
            c(i,j) = 3.0
         enddo
      enddo

!$xmp task on p(1)
!$xmp loop on t(j)
      do j=1, w
         do i=1, N
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task

!$xmp task on p(2:)
      if(procs .eq. 4) then
!$xmp loop on t(j)
         do j=w+1,N
            do i=1, N
               sa = sa+a(i,j)
               sb = sb+b(i,j)
               sc = sc+c(i,j)
            enddo
         enddo
      endif

!$xmp reduction(+:sa,sb,sc)
!$xmp end task

      result = 'OK'
      if(xmp_node_num().eq.1) then
         ans = w*1000
         if(sa.ne.ans) then
            result = 'NG'
            print *, 'sa',sa
         endif
         if(abs(sb-2.0*dble(ans)).gt.0.000000001) then
            result = 'NG'
         endif
         if(abs(sc-3*real(ans)).gt.0.000001) then
            result = 'NG'
         endif
      else
         ans = (N-w)*1000
         if(sa.ne.ans) then
            result = 'NG'
            print *, 'sa',sa
         endif
         if(abs(sb-2.0*dble(ans)).gt.0.000000001) then
            result = 'NG'
         endif
         if(abs(sc-3*real(ans)).gt.0.000001) then
            result = 'NG'
         endif
      endif

      print *, xmp_node_num(), 'testp073.f ', result
      end
