! testp021.f
! gmove指示文とasync節のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp template t4(N,N)
!$xmp template t5(N,N)
!$xmp template t6(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(block,cyclic) onto p
!$xmp distribute t3(cyclic,block) onto p
!$xmp distribute t4(cyclic,cyclic) onto p
!$xmp distribute t5(cyclic(2),cyclic(3)) onto p
!$xmp distribute t6(cyclic(4),cyclic(5)) onto p
      integer a1(N,N), a2(N,N), sa
      real*8  b1(N,N), b2(N,N), sb
      real*4  c1(N,N), c2(N,N), sc
!$xmp align a1(i,j) with t1(i,j)
!$xmp align b1(i,j) with t2(i,j)
!$xmp align c1(i,j) with t3(i,j)
!$xmp align a2(i,j) with t4(i,j)
!$xmp align b2(i,j) with t5(i,j)
!$xmp align c2(i,j) with t6(i,j)
      integer procs
      character(len=2) result

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            a1(i,j) = (j-1)*N+i
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            b1(i,j) = dble((j-1)*N+i)
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            c1(i,j) = real((j-1)*N+i)
         enddo
      enddo
!$xmp loop (i,j) on t4(i,j)
      do j=1, N
         do i=1, N
            a2(i,j) = 1
         enddo
      enddo
!$xmp loop (i,j) on t5(i,j)
      do j=1, N
         do i=1, N
            b2(i,j) = 1.0
         enddo
      enddo
!$xmp loop (i,j) on t6(i,j)
      do j=1, N
         do i=1, N
            c2(i,j) = 1.0
         enddo
      enddo

      
      procs = xmp_all_num_nodes()
      do j=1, procs
         if(j.eq.xmp_all_node_num()) then
!$xmp gmove in async(1)
            a1(:,:) = a2(:,:)
!$xmp gmove in async(2)
            b1(:,:) = b2(:,:)
!$xmp gmove in async(3)
            c1(:,:) = c2(:,:)
         endif
      enddo

!$xmp wait_async(1)
      if(xmp_node_num() .eq. 1) then
         a1(1,1) = a1(1,1)+10
      endif
!$xmp wait_async(2)
      if(xmp_node_num() .eq. 1) then
         b1(1,1) = b1(1,1)+30.0
      endif
!$xmp wait_async(3)
      if(xmp_node_num() .eq. 1) then
         c1(1,1) = c1(1,1)+20.0
      endif

      sa = 0
      sb = 0.0
      sc = 0.0
!$xmp loop on t1(i,j) reduction(+:sa)
      do j=1, N
         do i=1, N
            sa = sa+a1(i,j)
         enddo
      enddo
!$xmp loop on t2(i,j) reduction(+:sb)
      do j=1, N
         do i=1, N
            sb = sb+b1(i,j)
         enddo
      enddo
!$xmp loop on t3(i,j) reduction(+:sc)
      do j=1, N
         do i=1, N
            sc = sc+c1(i,j)
         enddo
      enddo

      result = 'OK'
      if(sa .ne. 1000010 .or.
     $     abs(sb-1000030.0) .gt. 0.000000001 .or.
     $     abs(sc-1000020.0) .gt. 0.0001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp021.f ', result
      
      end
