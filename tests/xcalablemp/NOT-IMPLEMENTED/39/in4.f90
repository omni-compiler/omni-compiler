! testp076.f
! task指示文、post指示文、wait指示文の組合せ

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      integer procs, half_procs, w
      integer a(N), sa
!$xmp align a(i) with t(i)
      character(len=2), result

      if(xmp_num_nodes().ne.4) then
         print *, 'You have to run this program by 4 nodes.'
      endif

!$xmp tasks
!$xmp task on p(1:2)
!$xmp loop on t(i)
      do i=1, N/2
         a(i) = i+N/2
      enddo
!$xmp post (p(3), 1)
!$xmp wait (p(3), 2)
!$xmp loop on t(i)
      do i=1, N/2
         a(i) = i
      enddo
!$xmp end task
!$xmp task on p(3:4)
!$xmp wait (p(1), 1)
!$xmp gmove in
      a(N/2+1:N) = a(1:N/2)
!$xmp post (p(1), 2)
!$xmp end task
!$xmp end tasks

      sa = 0
!$xmp loop on t(i)
      do i=1, N
         sa = sa+a(i)
      enddo

!$xmp reduction(+:sa)

      if(sa .eq. N*(N+1)/2) then
         result = 'OK'
      else
         print *, sa
         result = 'NG'
      endif
         
      print *, xmp_node_num(), 'testp076.f ', result

      end
