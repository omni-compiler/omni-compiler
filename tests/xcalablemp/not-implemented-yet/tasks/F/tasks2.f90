! testp077.f
! tasks指示文およびtask指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
      integer a(N,N), sa, ansa, lb, ub, procs, procs2
      real*8  b(N,N), sb, ansb
      real*4  c(N,N), sc, ansc
      character(len=2) result
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)
!$xmp shadow a(1,1)
!$xmp shadow b(2,2)
!$xmp shadow c(3,3)

      sa = 0
      sb = 0.0
      sc = 0.0
      
!$xmp loop on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = 2
            b(i,j) = 1.5
            c(i,j) = 0.5
         enddo
      enddo

      procs = xmp_num_nodes()
      procs2 = procs/4

!$xmp tasks
!$xmp task on p(1,1)
!$xmp loop on t(i,j)
      do j=1, N/procs2
         do i=1, N/4
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task
!$xmp task on p(1,2)
!$xmp loop on t(i,j)
      do j=N/procs2+1, N/procs2*2
         do i=1, N/4
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task
!$xmp task on p(2,1)
!$xmp loop on t(i,j)
      do j=1, N/procs2
         do i=N/4+1, N/2
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task
!$xmp task on p(2,2)
!$xmp loop on t(i,j)
      do j=N/procs2+1, N/procs2*2
         do i=N/4+1, N/2
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp end task
!$xmp end tasks

!$xmp reduction (+: sa, sb, sc)

      result = 'OK'
      ansa = (N*N/procs2)*2
      if(sa .ne. ansa) then
         result = 'NG'
      endif
      ansb = dble(N*N/procs2)*1.5
      if(abs(sb-ansb) .gt. 0.0000001) then
         result = 'NG'
      endif
      ansc = real(N*N/procs2)*0.5
      if(abs(sc-ansc) .gt. 0.00001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp077.f ', result

      end
