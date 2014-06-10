! testp110.f
! loop指示文とpost/wait指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N,N)
!$xmp distribute t(cyclic,cyclic) onto p
      integer a(N,N)
      integer ii, jj
!$xmp align a(i,j) with t(i,j)
      character(len=3) result

      if(xmp_num_nodes() .ne. 16) then
         print *, ' You have to run this program by 16 nodes.'
      endif
      
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = i
         enddo
      enddo

      ii = mod(xmp_node_num(), 4)+1
      jj = xmp_node_num()/4+1
!$xmp barrier
!$xmp loop (j) on t(:,j)
      do j=2, N
         if(j.ne.2) then
!$xmp wait(p(ii, mod(j-1,4)), 1
         endif
!$xmp gmove in
         a(:,j) = a(:,j-1)
!$xmp loop (i) on t(i,j)
         do i=1, N
            a(i,j) = a(i,j)+(j-1)*N
         enddo
         if(j.ne.N) then
!$xmp post(p(ii, mod(j+1,4)), 1
         endif
      enddo

      result = 'OK '
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            if(a(i,j) .ne. (j-1)*N+i) result = 'NG '
         enddo
      enddo

      print *, xmp_node_num(), 'testp110.f ', result

      end
      
