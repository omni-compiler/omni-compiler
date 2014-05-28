! testp022.f
! gmove指示文のテスト

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
!$xmp align a1(i) with t1(*,i)
!$xmp align b1(i) with t2(*,i)
!$xmp align c1(i) with t3(*,i)
!$xmp align a2(i) with t4(i,*)
!$xmp align b2(i) with t5(i,*)
!$xmp align c2(i) with t6(i,*)
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

      
      procs = xmp_num_nodes()
      do j=1, procs
         if(j.eq.xmp_node_num()) then
!$xmp gmove in
            a1(:,:) = a2(:,:)
!$xmp gmove in
            b1(:,:) = b2(:,:)
!$xmp gmove in
            c1(:,:) = c2(:,:)
         endif
      enddo

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
      if(sa .ne. 1000000 .or.
     $     abs(sb-1000000.0) .gt. 0.000000001 .or.
     $     abs(sc-1000000.0) .gt. 0.0001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp022.f ', result
      
      end
