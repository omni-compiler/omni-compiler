! testp087.f
! task指示文とbcast指示文の組み合わせ
      
      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
      integer procs, w
      integer a(N,N), sa, ansa
      real*8  b(N,N), sb, ansb
      real*4  c(N,N), sc, ansc
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)
      character(len=2) result

      if(xmp_num_nodes().ne.16) then
         print *, 'You have to run this program by 16 nodes.'
      endif

!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = (j-1)/10*N+(i-1)/10
            b(i,j) = dble((j-1)*N+(i-1))
            c(i,j) = real((j-1)*N+(i-1))
         enddo
      enddo

      sa = 0
      sb = 0.0
      sc = 0.0
      procs = xmp_num_nodes()
      if(mod(N,procs).eq.0) then
         w = N/procs
      else
         w = N/procs+1
      endif

!$xmp task on p(1:3,1:3)
!$xmp loop on t(j)
      do j=1, 3*w
         do i=1, 3*w
            sa = sa+a(i,j)
            sb = sa+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
      
!$xmp bcast (sa) from p(3,2) on p(1:3,1:3)
!$xmp bcast (sb) from p(3,3) on p(1:3,1:3)
!$xmp bcast (sc) from p(2,3) on p(1:3,1:3)
!$xmp end task

      result = 'OK'
      if(  xmp_node_num().eq.1 .or.
     $     xmp_node_num().eq.2 .or.
     $     xmp_node_num().eq.3 .or.
     $     xmp_node_num().eq.5 .or.
     $     xmp_node_num().eq.6 .or.
     $     xmp_node_num().eq.7 .or.
     $     xmp_node_num().eq.9 .or.
     $     xmp_node_num().eq.10 .or.
     $     xmp_node_num().eq.11 ) then
         ansa = 0
         ansb = 0.0
         ansc = 0.0
         do j=w+1, 2*w
            do i=2*w+1, 3*w
               ansa = ansa+(j-1)/10*N+(i-1)/10
            enddo
         enddo
         do j=2*w+1, 3*w
            do i=2*w+1, 3*w
               ansb = ansb+dble((j-1)*N+(i-1))
            enddo
         enddo
         do j=2*w+1, 3*w
            do i=w+1, 2*w
               ansc = ansc+real((j-1)*N+(i-1))
            enddo
         enddo
         if(sa.ne.ansa .or. sb.ne.ansb .or. sb.ne.ansc) then
            result = 'NG'
         endif
      else
         if(sa.ne.0 .or. sb.ne.0.0 .or. sc.ne.0.0) then
            result = 'NG'
         endif
      endif

      print *, xmp_node_num(), 'testp087.f ', result
      
      end
