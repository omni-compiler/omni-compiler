! testp088.f
! task指示文、post指示文、wait指示文の組合せ

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N,N)
!$xmp distribute t(cyclic,block) onto p
      integer a(N,N), sa
!$xmp align a(i,j) with t(i,j)
      character(len=2), result

      if(xmp_num_nodes().ne.16) then
         print *, 'You have to run this program by 16 nodes.'
      endif

!$xmp tasks
!$xmp task on p(:,1:2)
!$xmp loop (i,j) on t(i,j)
      do j=1, N/2
         do i=1, N
            a(i,j) = 2
         enddo
      enddo
!$xmp post (p(1,3), 1)
!$xmp wait (p(1,3), 2)
!$xmp loop (i,j) on t(i,j)
      do j=1, N/2
         do i=1, N
            a(i,j) = 1
         enddo
      enddo
!$xmp end task
!$xmp task on p(:,3:4)
!$xmp wait (p(1,1), 1)
!$xmp gmove in
      a(:,N/2+1:N) = a(:,1:N/2)
!$xmp post (p(1,1), 2)
!$xmp end task
!$xmp end tasks

      sa = 0
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            sa = sa+a(i,j)
         enddo
      enddo

!$xmp reduction(+:sa)

      if(sa.eq.1500000) then
         result = 'OK'
      else
         print *, sa
         result = 'NG'
      endif
         
      print *, xmp_node_num(), 'testp088.f ', result

      end
