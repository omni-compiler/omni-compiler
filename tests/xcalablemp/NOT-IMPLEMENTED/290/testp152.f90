! testp152.f
! bcast指示文のテスト：fromは無し、onは無し

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
      integer aa(N), a
      real*4 bb(N), b
      real*8 cc(N), c
      integer procs, procs2, ans
      character(len=2) result

      procs = xmp_num_nodes()
      procs2 = procs/4

      result = 'OK'
      a = xmp_node_num()
      b = real(a)
      c = dble(a)
      do i=1, N
         aa(i) = a+i-1
         bb(i) = real(a+i-1)
         cc(i) = dble(a+i-1)
      enddo
      
!$xmp bcast (a)
!$xmp bcast (b)
!$xmp bcast (c)
!$xmp bcast (aa)
!$xmp bcast (bb)
!$xmp bcast (cc)

      ans = 1
      if(a .ne. ans) result = 'NG'
      if(b .ne. real(ans)) result = 'NG'
      if(c .ne. dble(ans)) result = 'NG'
      do i=1, N
         if(aa(i) .ne. ans+i-1) result = 'NG'
         if(bb(i) .ne. real(ans+i-1)) result = 'NG'
         if(cc(i) .ne. dble(ans+i-1)) result = 'NG'
      enddo
      
      print *, xmp_node_num(), 'testp152.f ', result

      end
