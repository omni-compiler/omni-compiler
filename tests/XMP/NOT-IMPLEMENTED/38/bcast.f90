! testp075.f
! task指示文とbcast指示文の組み合わせ
      
      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      integer procs, w
      integer a(N,N), sa, ansa
      real*8  b(N,N), sb, ansb
      real*4  c(N,N), sc, ansc
!$xmp align a(*,i) with t(i)
!$xmp align b(i,*) with t(i)
!$xmp align c(*,i) with t(i)
      character(len=2) result

      if(xmp_num_nodes().lt.4) then
         print *, 'You have to run this program by more than 4 nodes.'
      endif

!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            a(i,j) = (j-1)/10*N+(i-1)/10
            c(i,j) = real((j-1)*N+(i-1))
         enddo
      enddo

      do j=1, N
!$xmp loop on t(i)
         do i=1, N
            b(i,j) = dble((j-1)*N+(i-1))
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

!$xmp task on p(2:3)
!$xmp loop on t(j)
      do j=w+1,2*w
         do i=1, N
            sa = sa+a(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
      do j=1,N
!$xmp loop on t(i)
         do i=w+1, 3*w
            sb = sa+b(i,j)
         enddo
      enddo

!$xmp bcast (sa) from p(3)
!$xmp bcast (sb) from p(2)
!$xmp bcast (sc) from p(3)
!$xmp end task

      result = 'OK'
      if(xmp_node_num().eq.2 .or. xmp_node_num().eq.3) then
         ansa = 0
         ansb = 0.0
         ansc = 0.0
         do j=2*w+1, 3*w
            do i=1, N
               ansa = ansa+(j-1)/10*N+(i-1)/10
               ansc = ansc+real((j-1)*N+(i-1))
            enddo
         enddo
         do j=1, N
            do i=w+1, 2*w
               ansb = ansb+dble((j-1)*N+(i-1))
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

      print *, xmp_node_num(), 'testp075.f ', result
      
      end
