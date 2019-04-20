! testp115.f
! barrier指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      integer a(N), procs, w
!$xmp align a(i) with t(i)
      character(len=3) result

      procs = xmp_num_nodes()
      if(mod(N,procs) .eq. 0) then
         w = N/procs
      else
         w = N/procs+1
      endif
      
      do j=2, procs
         if(xmp_node_num() .eq. 1) then
            do i=1, w
               a(i) = (j-1)*w+i
            enddo
         endif
!$xmp barrier on p(:)
         if(xmp_node_num() .eq. j) then
!$xmp gmove in
            a((j-1)*w+1:j*w) = a(1:w)
         endif
      enddo

      if(xmp_node_num() .eq. 1) then
         do i=1, w
            a(i) = i
         enddo
      endif
      
      result = 'OK '
!$xmp loop on t(i)
      do i=1, N
         if(a(i) .ne. i) result = 'NG '
      enddo

      print *, xmp_node_num(), 'testp115.f ', result

      end
