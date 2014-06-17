! testp120.f
! barrier指示文のテスト (部分ノード)

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
      integer a(N,N), procs, w1, w2, p1, p2, pi, pj
!$xmp align a(i,j) with t(i,j)
      character(len=3) result

      procs = xmp_num_nodes()
      p1 = 4
      p2 = procs/4
      w1 = 250
      if(mod(N,p2) .eq. 0) then
         w2 = N/p2
      else
         w2 = N/p2+1
      endif
      
      do k=1, procs-1
         pi = mod(k,4)
         pj = k/4
         
         if(xmp_node_num() .eq. 1) then
            do j=1, w2
               do i=1, w1
                  a(i,j) = (j-1+pj*w2)*N+pi*w1+i
               enddo
            enddo
         endif

!$xmp barrier on p(1:pi+1,1:pj+1)

         if(xmp_node_num() .eq. k+1) then
!$xmp gmove in
            a(pi*w1+1:(pi+1)*w1,pj*w2+1:(pj+1)*w2) = a(1:w1,1:w2)
         endif
      enddo

      if(xmp_node_num() .eq. 1) then
         do j=1, w2
            do i=1, w1
               a(i,j) = (j-1)*N+i
            enddo
         enddo
      endif
      
      result = 'OK '
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            if(a(i,j) .ne. (j-1)*N+i) result = 'NG '
         enddo
      enddo

      print *, xmp_node_num(), 'testp120.f ', result

      end
