! testp127.f
! bcast指示文のテスト：fromはnode-ref、onはなし

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
      integer aa(N), a
      real*4 bb(N), b
      real*8 cc(N), c
      integer procs
      character(len=2) result

      procs = xmp_num_nodes()

      result = 'OK'
      do j=1, procs
         a = xmp_node_num()
         b = real(a)
         c = dble(a)
         do i=1, N
            aa(i) = a+i-1
            bb(i) = real(a+i-1)
            cc(i) = dble(a+i-1)
         enddo

!$xmp bcast (a) from p(j)
!$xmp bcast (b) from p(j)
!$xmp bcast (c) from p(j)
!$xmp bcast (aa) from p(j)
!$xmp bcast (bb) from p(j)
!$xmp bcast (cc) from p(j)

         if(a .ne. j) result = 'NG'
         if(b .ne. real(j)) result = 'NG'
         if(c .ne. dble(j)) result = 'NG'
         do i=1, N
            if(aa(i) .ne. j+i-1) result = 'NG'
            if(bb(i) .ne. real(j+i-1)) result = 'NG'
            if(cc(i) .ne. dble(j+i-1)) result = 'NG'
         enddo
      enddo

      print *, xmp_node_num(), 'testp127.f ', result

      end
