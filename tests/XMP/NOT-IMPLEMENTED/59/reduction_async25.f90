! testp084.f
! task指示文とreduction指示文の組合せ

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N)
!$xmp distribute t(gblock((/50,50,50,850/),block)) onto p
      integer procs
      integer a(N,N), sa, ans
      real*8  b(N,N), sb
      real*4  c(N,N), sc
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)
      character(len=2) result

      if(xmp_num_nodes().ne.16) then
         print *, 'You have to run this program by 16 nodes.'
      endif

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

      procs = xmp_num_nodes()

!$xmp task on p(1,1)
!$xmp loop (i,j) on t(i,j)
      do j=1, 250
         do i=1, 50
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task

!$xmp task on p(2:4,2:4)
      if(procs .eq. 16) then
!$xmp loop (i,j) on t(i,j)
         do j=751,N
            do i=151, N
               sa = sa+a(i,j)
               sb = sb+b(i,j)
               sc = sc+c(i,j)
            enddo
         enddo
!$xmp reduction(+:sa,sb,sc) async(1)
!$xmp wait_async(1)
         sa = sa+1
         sb = sb+1.0
         sc = sc+1.0
      endif
!$xmp end task

      result = 'OK'
      if(xmp_node_num().eq.1) then
         ans = 12500
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
      else if(xmp_node_num().eq.6 .or.
     $        xmp_node_num().eq.7 .or.
     $        xmp_node_num().eq.8 .or.
     $        xmp_node_num().eq.10 .or.
     $        xmp_node_num().eq.11 .or.
     $        xmp_node_num().eq.12 .or.
     $        xmp_node_num().eq.14 .or.
     $        xmp_node_num().eq.15 .or.
     $        xmp_node_num().eq.16 ) then
         ans = 212501
         if(sa.ne.ans) then
            result = 'NG'
            print *, 'sa',sa
         endif
         if(abs(sb-2.0*dble(ans)).gt.0.0000001) then
            result = 'NG'
         endif
         if(abs(sc-3*real(ans)).gt.0.0001) then
            result = 'NG'
         endif
      else
         if(sa.ne.0.or.sb.ne.0.0.or.sc.ne.0.0) then
            result = 'NG'
         endif   
      endif

      print *, xmp_node_num(), 'testp084.f ', result
      end

         
