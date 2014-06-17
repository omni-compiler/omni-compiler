! testp133.f
! bcast指示文のテスト：fromは無し、onはnode-ref

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
      a = xmp_node_num()
      b = real(a)
      c = dble(a)
      do i=1, N
         aa(i) = a+i-1
         bb(i) = real(a+i-1)
         cc(i) = dble(a+i-1)
      enddo
      
!$xmp bcast (a) on p(:)
!$xmp bcast (b) on p(:)
!$xmp bcast (c) on p(:)
!$xmp bcast (aa) on p(:)
!$xmp bcast (bb) on p(:)
!$xmp bcast (cc) on p(:)

      j = 1
      if(a .ne. j) result = 'NG'
      if(b .ne. real(j)) result = 'NG'
      if(c .ne. dble(j)) result = 'NG'
      do i=1, N
         if(aa(i) .ne. j+i-1) result = 'NG'
         if(bb(i) .ne. real(j+i-1)) result = 'NG'
         if(cc(i) .ne. dble(j+i-1)) result = 'NG'
      enddo
      
      print *, xmp_node_num(), 'testp133.f ', result

      end
